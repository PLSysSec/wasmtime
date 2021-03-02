//! A pass over Cranelift IR which implements the Blade algorithm

use crate::cursor::{Cursor, EncCursor};
use crate::entity::{EntitySet, SecondaryMap};
use crate::flowgraph::ControlFlowGraph;
use crate::ir::{condcodes::IntCC, dfg::Bounds, ArgumentPurpose, Function, Inst, InstBuilder, InstructionData, Opcode, Value, ValueDef};
use crate::isa::TargetIsa;
use crate::settings::{BladeType, SwitchbladeCallconv};
use rs_graph::linkedlistgraph::{Edge, LinkedListGraph, Node};
use rs_graph::maxflow::pushrelabel::PushRelabel;
use rs_graph::maxflow::MaxFlow;
use rs_graph::traits::{Directed, GraphSize};
use rs_graph::Buildable;
use rs_graph::Builder;
use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use alloc::vec::Vec;

/// If this is `true`, then Blade will use a fake bound information for any
/// address which does not have associated bounds information.
pub(crate) const ALLOW_FAKE_SLH_BOUNDS: bool = true;

/// Nonsense array length (in bytes) to use as the fake bound information (see
/// `ALLOW_FAKE_SLH_BOUNDS` above)
pub(crate) const FAKE_SLH_ARRAY_LENGTH_BYTES: u32 = 2345;

/// Dump (to stdout) the Cranelift IR for each function before this pass
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_FUNCTION_BEFORE_BLADE`
const DEBUG_PRINT_FUNCTION_BEFORE: bool = false;

/// Dump (to stdout) the Cranelift IR for each function after this pass
///
/// Note that the fences Blade inserts are not (currently) visible in the dumped
/// Cranelift IR; but the SLHs are
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_FUNCTION_AFTER_BLADE`
const DEBUG_PRINT_FUNCTION_AFTER: bool = false;

/// Print the detailed location of each fence/SLH as we insert it
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_BLADE_DEF_LOCATIONS`
const DEBUG_PRINT_DETAILED_DEF_LOCATIONS: bool = false;

/// Print the list of all BC-tainted nodes in the function.
/// This only has an effect under Switchblade strategies.
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_SWITCHBLADE_BC_NODES`
const DEBUG_PRINT_BC_NODES: bool = false;

/// Print the list of all sources in the Blade graph.
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_BLADE_SOURCES`
const DEBUG_PRINT_SOURCES: bool = false;

/// Print the list of all sinks in the Blade graph.
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_BLADE_SINKS`
const DEBUG_PRINT_SINKS: bool = false;

/// Should we print various statistics about Blade's actions
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `PRINT_BLADE_STATS`
const PRINT_BLADE_STATS: bool = false;

/// Should we dump various statistics about Blade's actions, in JSON form, to a
/// file blade_stats/<func_name>.json
///
/// Even if this is set to `false` here, you can enable it by setting the
/// environment variable `DUMP_BLADE_STATS`
const DUMP_BLADE_STATS: bool = false;

pub fn do_blade(func: &mut Function, isa: &dyn TargetIsa, cfg: &ControlFlowGraph) {
    let blade_type = isa.flags().blade_type();
    if blade_type == BladeType::None {
        return;
    }

    if DEBUG_PRINT_FUNCTION_BEFORE || std::env::var("PRINT_FUNCTION_BEFORE_BLADE").is_ok() {
        println!("Function before blade:\n{}", func.display(isa));
    }

    let stats = BladePass::new(func, cfg, isa).run();

    if DEBUG_PRINT_FUNCTION_AFTER || std::env::var("PRINT_FUNCTION_AFTER_BLADE").is_ok() {
        println!("Function after blade:\n{}", func.display(isa));
    }

    if PRINT_BLADE_STATS || std::env::var("PRINT_BLADE_STATS").is_ok() {
        match blade_type {
            BladeType::Lfence | BladeType::LfencePerBlock | BladeType::BaselineFence | BladeType::SwitchbladeFenceA | BladeType::SwitchbladeFenceB | BladeType::SwitchbladeFenceC => {
                println!("function {}:\n  inserted {} (static) lfences", func.name, stats.static_fences_inserted);
                if is_switchblade(&blade_type) {
                    println!("  ({} due to BC calling conventions)", stats.static_fences_inserted_due_to_bc_calling_conventions);
                }
                assert_eq!(stats.static_slhs_inserted, 0);
            }
            BladeType::Slh | BladeType::BaselineSlh => {
                println!("function {}:\n  inserted {} (static) SLHs", func.name, stats.static_slhs_inserted);
                assert_eq!(stats.static_fences_inserted, 0);
            }
            BladeType::None => {
                assert_eq!(stats.static_fences_inserted, 0);
                assert_eq!(stats.static_slhs_inserted, 0);
            }
        }
        println!("  number of sources in the Blade graph: {}", stats.num_sources);
        println!("  number of sinks in the Blade graph: {}", stats.num_sinks);
    }

    if DUMP_BLADE_STATS || std::env::var("DUMP_BLADE_STATS").is_ok() {
        let json_object = json::object! {
            static_fences_inserted: stats.static_fences_inserted,
            static_slhs_inserted: stats.static_slhs_inserted,
            num_sources: stats.num_sources,
            num_sinks: stats.num_sinks,
            static_fences_inserted_due_to_bc_calling_conventions: stats.static_fences_inserted_due_to_bc_calling_conventions,
        };
        use std::path::Path;
        let dir = Path::new("blade_stats");
        std::fs::create_dir_all(dir).unwrap();
        let filename = format!("{}.json", func.name);
        let filepath = dir.join(filename);
        use std::io::Write;
        let mut f = std::fs::File::create(filepath).unwrap();
        writeln!(f, "{:#}", json_object).unwrap();
    }
}

/// Returns `true` if the blade type is one of the Switchblade types
fn is_switchblade(blade_type: &BladeType) -> bool {
    match blade_type {
        BladeType::SwitchbladeFenceA => true,
        BladeType::SwitchbladeFenceB => true,
        BladeType::SwitchbladeFenceC => true,
        _ => false,
    }
}

/// All of the data we need to perform the Blade pass
struct BladePass<'a> {
    /// The function which we're performing the Blade pass on
    func: &'a mut Function,
    /// The CFG for that function
    cfg: &'a ControlFlowGraph,
    /// The `TargetIsa`
    isa: &'a dyn TargetIsa,
    /// The `BladeType` of the pass we're doing
    blade_type: BladeType,
    /// Whether we are also protecting from v1.1 (as opposed to just v1)
    blade_v1_1: bool,
    /// the Switchblade calling convention. (Not used if `blade_type` isn't Switchblade.)
    switchblade_callconv: SwitchbladeCallconv,
    /// The def-use graph for that function, with no call edges (see
    /// documentation on `DefUseGraph`). `SimpleCache` ensures that we compute
    /// this a maximum of once, and only if it is needed
    def_use_graph_no_call_edges: SimpleCache<DefUseGraph>,
    /// The def-use graph for that funcntion, with call edges (see
    /// documentation on `DefUseGraph`). `SimpleCache` ensures that we compute
    /// this a maximum of once, and only if it is needed
    def_use_graph_with_call_edges: SimpleCache<DefUseGraph>,
    /// The `BCData` for that function. `SimpleCache` ensures that we compute
    /// this a maximum of once, and only if it is needed
    bcdata: SimpleCache<BCData>,
}

impl<'a> BladePass<'a> {
    pub fn new(func: &'a mut Function, cfg: &'a ControlFlowGraph, isa: &'a dyn TargetIsa) -> Self {
        Self {
            func,
            cfg,
            isa,
            blade_type: isa.flags().blade_type(),
            blade_v1_1: isa.flags().blade_v1_1(),
            switchblade_callconv: isa.flags().switchblade_callconv(),
            def_use_graph_no_call_edges: SimpleCache::new(),
            def_use_graph_with_call_edges: SimpleCache::new(),
            bcdata: SimpleCache::new(),
        }
    }

    /// Returns `true` if the blade type is one of the Switchblade types
    fn is_switchblade(&self) -> bool {
        is_switchblade(&self.blade_type)
    }

    fn get_def_use_graph_no_call_edges(&self) -> Ref<DefUseGraph> {
        self.def_use_graph_no_call_edges
            .get_or_insert_with(|| DefUseGraph::for_function(self.func, self.cfg, false))
    }

    fn get_def_use_graph_with_call_edges(&self) -> Ref<DefUseGraph> {
        self.def_use_graph_with_call_edges
            .get_or_insert_with(|| DefUseGraph::for_function(self.func, self.cfg, true))
    }

    fn get_bcdata(&self) -> Ref<BCData> {
        let dug_no_call_edges = self.get_def_use_graph_no_call_edges();
        let dug_with_call_edges = self.get_def_use_graph_with_call_edges();
        self.bcdata.get_or_insert_with(move || BCData::new(
            self.func,
            self.blade_type,
            self.switchblade_callconv,
            &dug_no_call_edges,
            &dug_with_call_edges
        ))
    }

    /// Run the Blade pass, inserting fences/SLHs as necessary, and return the
    /// `BladeStats` with statistics about Blade's actions
    pub fn run(&mut self) -> BladeStats {
        let mut stats = BladeStats::new();

        if self.is_switchblade() {
            // Preliminary pass to insert fences to ensure that function
            // arguments and/or return values aren't BC (depending on the
            // Switchblade calling convention)
            let mut insts_needing_fences = vec![];
            let bcdata = self.get_bcdata();
            if DEBUG_PRINT_BC_NODES || std::env::var("PRINT_SWITCHBLADE_BC_NODES").is_ok() {
                println!("BC-tainted nodes: {:?}", bcdata.tainted_values.keys().filter(|&val| bcdata.tainted_values.contains(val)).collect::<Vec<Value>>());
            }
            for block in self.func.layout.blocks() {
                for inst in self.func.layout.block_insts(block) {
                    let opcode = self.func.dfg[inst].opcode();
                    if opcode.is_call() {
                        // if the calling convention is that call arguments must
                        // not be BC, fence before the call if any arguments
                        // would be BC
                        match self.switchblade_callconv {
                            SwitchbladeCallconv::NotNot | SwitchbladeCallconv::NotMay => {
                                if self.func.dfg.inst_args(inst).iter().any(|&arg| bcdata.tainted_values.contains(arg)) {
                                    insts_needing_fences.push(inst);
                                }
                            }
                            SwitchbladeCallconv::MayNot | SwitchbladeCallconv::MayMay => (),
                        }
                    }
                    if opcode.is_return() {
                        // if the calling convention is that return values must
                        // not be BC, fence before the return if any returned
                        // values would be BC
                        match self.switchblade_callconv {
                            SwitchbladeCallconv::NotNot | SwitchbladeCallconv::MayNot => {
                                if self.func.dfg.inst_args(inst).iter().any(|&arg| bcdata.tainted_values.contains(arg)) {
                                    insts_needing_fences.push(inst);
                                }
                            }
                            SwitchbladeCallconv::NotMay | SwitchbladeCallconv::MayMay => (),
                        }
                    }
                }
            }
            std::mem::drop(bcdata);  // drops the borrow of self, so we can borrow it mutably to insert fences
            let mut fence_inserter = FenceInserter::new(self.func, self.blade_type, &mut stats);
            for inst in insts_needing_fences {
                fence_inserter.insert_fence_before(&BladeNode::Sink(inst));
            }
            // so far, all the fences we've inserted are due to BC calling conventions; and we
            // won't insert any more due to BC calling conventions
            stats.static_fences_inserted_due_to_bc_calling_conventions = stats.static_fences_inserted;
        }

        // build the Blade graph
        let store_values_are_sinks =
            if self.blade_type == BladeType::Slh && self.blade_v1_1 {
                // For SLH to protect from v1.1, store values must be marked as sinks
                true
            } else {
                false
            };
        let sources = match (self.blade_type, self.blade_v1_1) {
            (BladeType::SwitchbladeFenceA, false) => Sources::BCAddr,
            (BladeType::SwitchbladeFenceA, true) => unimplemented!("switchblade_fence_a is not implemented for v1.1 yet"),
            (BladeType::SwitchbladeFenceB, false) => Sources::BCAddr,
            (BladeType::SwitchbladeFenceB, true) => unimplemented!("switchblade_fence_b is not implemented for v1.1 yet"),
            (BladeType::SwitchbladeFenceC, false) => Sources::BCAddr,
            (BladeType::SwitchbladeFenceC, true) => unimplemented!("switchblade_fence_c is not implemented for v1.1 yet"),
            (_, false) => Sources::NonConstantAddr,
            (_, true) => Sources::AllLoads,
        };
        let blade_graph = self.build_blade_graph(store_values_are_sinks, sources);

        if DEBUG_PRINT_SOURCES || std::env::var("PRINT_BLADE_SOURCES").is_ok() {
            let sources: Vec<Value> = blade_graph.source_values().collect();
            println!("Blade sources: {:?}", sources);
        }
        if DEBUG_PRINT_SINKS || std::env::var("PRINT_BLADE_SINKS").is_ok() {
            let sinks: Vec<&BladeNode> = blade_graph.sink_bladenodes().collect();
            println!("Blade sinks: {:?}", sinks);
        }

        stats.num_sources = blade_graph.source_nodes().count();
        stats.num_sinks = blade_graph.sink_nodes().count();

        // insert the fences / SLHs
        match self.blade_type {
            BladeType::BaselineFence => {
                let mut fence_inserter = FenceInserter::new(self.func, self.blade_type, &mut stats);
                for source in blade_graph.source_nodes() {
                    // insert a fence after every source
                    fence_inserter.insert_fence_after(&blade_graph.node_to_bladenode_map[&source]);
                }
            }
            BladeType::BaselineSlh => {
                let mut slh_inserter = SLHInserter::new(self.func, self.isa, &mut stats);
                for source in blade_graph.source_nodes() {
                    // use SLH on every source
                    slh_inserter.apply_slh_to_bnode(&blade_graph.node_to_bladenode_map[&source]);
                }
            }
            BladeType::Lfence | BladeType::LfencePerBlock | BladeType::SwitchbladeFenceA | BladeType::SwitchbladeFenceB | BladeType::SwitchbladeFenceC => {
                let mut fence_inserter = FenceInserter::new(self.func, self.blade_type, &mut stats);
                for cut_edge in blade_graph.min_cut() {
                    let edge_src = blade_graph.graph.src(cut_edge);
                    let edge_snk = blade_graph.graph.snk(cut_edge);
                    if edge_src == blade_graph.source_node {
                        // source -> n : fence after n
                        fence_inserter.insert_fence_after(&blade_graph.node_to_bladenode_map[&edge_snk]);
                    } else if edge_snk == blade_graph.sink_node {
                        // n -> sink : fence before (def of) n
                        fence_inserter.insert_fence_before(&blade_graph.node_to_bladenode_map[&edge_src]);
                    } else {
                        // n -> m : fence before m
                        fence_inserter.insert_fence_before(&blade_graph.node_to_bladenode_map[&edge_snk]);
                    }
                }
            }
            BladeType::Slh => {
                let mut slh_inserter = SLHInserter::new(self.func, self.isa, &mut stats);
                for cut_edge in blade_graph.min_cut() {
                    let edge_src = blade_graph.graph.src(cut_edge);
                    let edge_snk = blade_graph.graph.snk(cut_edge);
                    if edge_src == blade_graph.source_node {
                        // source -> n : apply SLH to the instruction that produces n
                        slh_inserter.apply_slh_to_bnode(&blade_graph.node_to_bladenode_map[&edge_snk]);
                    } else if edge_snk == blade_graph.sink_node {
                        // n -> sink : for SLH we can't cut at n (which is a sink instruction), we have
                        // to trace back through the graph and cut at all sources which lead to n
                        for node in blade_graph.ancestors_of(edge_src) {
                            slh_inserter.apply_slh_to_bnode(&blade_graph.node_to_bladenode_map[&node]);
                        }
                    } else {
                        // n -> m : likewise, apply SLH to all sources which lead to n
                        for node in blade_graph.ancestors_of(edge_src) {
                            slh_inserter.apply_slh_to_bnode(&blade_graph.node_to_bladenode_map[&node]);
                        }
                    }
                }
            }
            BladeType::None => panic!("Shouldn't reach here with Blade setting None"),
        }

        stats
    }

    /// `store_values_are_sinks`: if `true`, then the value operand to a store
    /// instruction is considered a sink. if `false`, it is not.
    /// For instance in the instruction "store x to addrA", if
    /// `store_values_are_sinks` is `true`, then both `x` and `addrA` are sinks,
    /// but if it is `false`, then just `addrA` is a sink.
    fn build_blade_graph(&self, store_values_are_sinks: bool, sources: Sources) -> BladeGraph {
        let mut builder = BladeGraphBuilder::with_nodes_for_func(self.func);

        // first we add edges for actual data dependencies
        // for instance in the following pseudocode:
        //     x = load y
        //     z = x + 2
        //     branch on z
        // we need an edge x -> z; that's what we're doing now
        // later we will add other edges to mark sinks and sources
        // (in this example, z -> sink and source -> x)
        for val in self.func.dfg.values() {
            let node = builder.bladenode_to_node_map[&BladeNode::ValueDef(val)]; // must exist
            // This step uses the `def_use_graph_no_call_edges`, which means
            // that even if a call argument is tainted (transient), the result
            // of the call will not be tainted. (In fact, results of call
            // instructions will never be tainted, because there will be no
            // incoming edges.)
            // This is ok because return values are prevented from being tainted
            // (required to be stable) by fencing in the callee if necessary, so
            // the caller (us) may safely assume the return value is untainted
            // (stable).
            let def_use_graph = self.get_def_use_graph_no_call_edges();
            for val_use in def_use_graph.uses_of_val(val) {
                match *val_use {
                    ValueUse::Inst(inst_use) => {
                        // add an edge from val to the result of inst_use
                        // TODO this assumes that all results depend on all operands;
                        // are there any instructions where this is not the case for our purposes?
                        for &result in self.func.dfg.inst_results(inst_use) {
                            builder.add_edge_from_node_to_value(node, result);
                        }
                    }
                    ValueUse::Value(val_use) => {
                        // add an edge from val to val_use
                        builder.add_edge_from_node_to_value(node, val_use);
                    }
                }
            }
        }

        // now we find sources and sinks, and add edges to/from our global source and sink nodes
        for block in self.func.layout.blocks() {
            for inst in self.func.layout.block_insts(block) {
                if self.blade_type == BladeType::SwitchbladeFenceC && is_there_a_fence_at_top_of_this_linear_block(self.func, inst) {
                    // SwitchbladeFenceC doesn't mark anything as a source or a
                    // sink if there is a fence at the top of its linear block
                    continue;
                }

                let idata = &self.func.dfg[inst];
                let op = idata.opcode();
                if op.can_load() {
                    // loads may be sources (their loaded values) and/or sinks (their addresses)

                    // handle load as a source, if necessary
                    let load_is_a_source = match sources {
                        Sources::AllLoads => true,
                        Sources::NonConstantAddr => !load_is_constant_addr(self.func, inst),
                        Sources::BCAddr => {
                            // address is BC if any of its components are
                            let bcdata = self.get_bcdata();
                            self.func.dfg.inst_args(inst).iter().any(|&val| bcdata.tainted_values.contains(val))
                        }
                    };
                    if load_is_a_source {
                        for &result in self.func.dfg.inst_results(inst) {
                            builder.mark_as_source(result);
                        }
                    }

                    // handle load as a sink, except for fills
                    if !(op == Opcode::Fill || op == Opcode::FillNop) {
                        let inst_sink_node = builder.add_sink_node_for_inst(inst);
                        // for each address component variable of inst,
                        // add edge address_component_variable_node -> sink
                        // XXX X86Pop has an implicit dependency on %rsp which is not captured here
                        for &arg in self.func.dfg.inst_args(inst) {
                            builder.add_edge_from_value_to_node(arg, inst_sink_node);
                        }
                    }

                } else if op.can_store() {
                    // loads are both sources and sinks, but stores are just sinks

                    let inst_sink_node = builder.add_sink_node_for_inst(inst);
                    // similar to the load case above, but special treatment for the value being stored
                    // XXX X86Push has an implicit dependency on %rsp which is not captured here
                    if store_values_are_sinks {
                        for &arg in self.func.dfg.inst_args(inst) {
                            builder.add_edge_from_value_to_node(arg, inst_sink_node);
                        }
                    } else {
                        // SC: as far as I can tell, all stores (that have arguments) always
                        //   have the value being stored as the first argument
                        //   and everything after is address args
                        for &arg in self.func.dfg.inst_args(inst).iter().skip(1) { // skip the first argument
                            builder.add_edge_from_value_to_node(arg, inst_sink_node);
                        }
                    };

                } else if op.is_branch() {
                    // conditional branches are sinks

                    let inst_sink_node = builder.add_sink_node_for_inst(inst);

                    // blade only does conditional branches but this will handle indirect jumps as well
                    // `inst_fixed_args` gets the condition args for branches,
                    //   and ignores the destination block params (which are also included in args)
                    for &arg in self.func.dfg.inst_fixed_args(inst) {
                        builder.add_edge_from_value_to_node(arg, inst_sink_node);
                    }

                }
                if op.is_call() {
                    // to avoid interprocedural analysis, we require that function
                    // arguments are stable, so we mark arguments to a call as sinks
                    let inst_sink_node = builder.add_sink_node_for_inst(inst);
                    for &arg in self.func.dfg.inst_args(inst) {
                        builder.add_edge_from_value_to_node(arg, inst_sink_node);
                    }
                }
                if op.is_return() {
                    // to avoid interprocedural analysis, we require that function
                    // return values are stable, so we mark return values as sinks
                    let inst_sink_node = builder.add_sink_node_for_inst(inst);
                    for &arg in self.func.dfg.inst_args(inst) {
                        builder.add_edge_from_value_to_node(arg, inst_sink_node);
                    }
                }
            }
        }

        // we no longer mark function parameters as transient, since we require that
        // they are stable on the caller side (so this is commented)
        /*
        let entry_block = self.func
            .layout
            .entry_block()
            .expect("Failed to find entry block");
        for &func_param in self.func.dfg.block_params(entry_block).iter().skip(1) {
            // parameters of the entry block == parameters of the function
            // the skip(1) is because the first param is what Cranelift uses to
            // supply the linear memory base address, it's not actually a Wasm
            // function parameter
            builder.mark_as_source(func_param);
        }
        */

        builder.build()
    }
}

struct BladeStats {
    /// Total number of static fences inserted by Blade
    static_fences_inserted: usize,
    /// Total number of static SLHs inserted by Blade
    static_slhs_inserted: usize,
    /// Total number of source nodes in the Blade graph
    num_sources: usize,
    /// Total number of sink nodes in the Blade graph
    num_sinks: usize,
    /// How many of Blade's static fences were inserted during the BC-calling-conventions pass
    static_fences_inserted_due_to_bc_calling_conventions: usize,
}

impl BladeStats {
    fn new() -> Self {
        Self {
            static_fences_inserted: 0,
            static_slhs_inserted: 0,
            num_sources: 0,
            num_sinks: 0,
            static_fences_inserted_due_to_bc_calling_conventions: 0,
        }
    }
}

struct FenceInserter<'a> {
    /// The function we're inserting fences into
    func: &'a mut Function,
    /// the `BladeType`
    blade_type: BladeType,
    /// the `BladeStats` where we record how many fences we're inserting
    stats: &'a mut BladeStats,
    /// if `true`, then we print (to stdout) the location of each fence as we insert it
    verbose: bool,
}

impl<'a> FenceInserter<'a> {
    fn new(func: &'a mut Function, blade_type: BladeType, stats: &'a mut BladeStats) -> Self {
        Self {
            func,
            blade_type,
            stats,
            verbose: DEBUG_PRINT_DETAILED_DEF_LOCATIONS || std::env::var("PRINT_BLADE_DEF_LOCATIONS").is_ok(),
        }
    }

    fn insert_fence_before(&mut self, bnode: &BladeNode) {
        match bnode {
            BladeNode::ValueDef(val) => match self.func.dfg.value_def(*val) {
                ValueDef::Result(inst, _) => {
                    match self.blade_type {
                        BladeType::Lfence | BladeType::BaselineFence | BladeType::SwitchbladeFenceA => {
                            // cut at this value by putting lfence before `inst`
                            insert_fence_before_inst(self.func, inst, self.stats, self.verbose);
                        }
                        BladeType::LfencePerBlock | BladeType::SwitchbladeFenceB | BladeType::SwitchbladeFenceC => {
                            // just put one fence at the beginning of the block.
                            // this stops speculation due to branch mispredictions.
                            insert_fence_at_beginning_of_block(self.func, inst, self.stats, self.verbose);
                        }
                        _ => panic!(
                            "This function didn't expect to be called with blade_type {:?}",
                            self.blade_type
                        ),
                    }
                }
                ValueDef::Param(block, _) => {
                    // cut at this value by putting lfence at beginning of
                    // the `block`, that is, before the first instruction
                    let first_inst = self.func
                        .layout
                        .first_inst(block)
                        .expect("block has no instructions");
                    insert_fence_before_inst(self.func, first_inst, self.stats, self.verbose);
                }
            },
            BladeNode::Sink(inst) => {
                match self.blade_type {
                    BladeType::Lfence | BladeType::BaselineFence | BladeType::SwitchbladeFenceA => {
                        // cut at this instruction by putting lfence before it
                        insert_fence_before_inst(self.func, *inst, self.stats, self.verbose);
                    }
                    BladeType::LfencePerBlock | BladeType::SwitchbladeFenceB | BladeType::SwitchbladeFenceC => {
                        // just put one fence at the beginning of the block.
                        // this stops speculation due to branch mispredictions.
                        insert_fence_at_beginning_of_block(self.func, *inst, self.stats, self.verbose);
                    }
                    _ => panic!(
                        "This function didn't expect to be called with blade_type {:?}",
                        self.blade_type
                    ),
                }
            }
        }
    }

    fn insert_fence_after(&mut self, bnode: &BladeNode) {
        match bnode {
            BladeNode::ValueDef(val) => match self.func.dfg.value_def(*val) {
                ValueDef::Result(inst, _) => {
                    match self.blade_type {
                        BladeType::Lfence | BladeType::BaselineFence | BladeType::SwitchbladeFenceA => {
                            // cut at this value by putting lfence after `inst`
                            insert_fence_after_inst(self.func, inst, self.stats, self.verbose);
                        }
                        BladeType::LfencePerBlock | BladeType::SwitchbladeFenceB | BladeType::SwitchbladeFenceC => {
                            // just put one fence at the beginning of the block.
                            // this stops speculation due to branch mispredictions.
                            insert_fence_at_beginning_of_block(self.func, inst, self.stats, self.verbose);
                        }
                        _ => panic!(
                            "This function didn't expect to be called with blade_type {:?}",
                            self.blade_type
                        ),
                    }
                }
                ValueDef::Param(block, _) => {
                    // cut at this value by putting lfence at beginning of
                    // the `block`, that is, before the first instruction
                    let first_inst = self.func
                        .layout
                        .first_inst(block)
                        .expect("block has no instructions");
                    insert_fence_before_inst(self.func, first_inst, self.stats, self.verbose);
                }
            },
            BladeNode::Sink(_) => panic!("Fencing after a sink instruction"),
        }
    }
}

/// Primitive that literally inserts a fence before an `Inst`.
fn insert_fence_before_inst(func: &mut Function, inst: Inst, stats: &mut BladeStats, verbose: bool) {
    if func.pre_lfence[inst] {
        // do nothing, already had a fence here
    } else {
        if verbose {
            println!("inserting fence before instruction {:?}", func.dfg[inst]);
        }
        func.pre_lfence[inst] = true;
        stats.static_fences_inserted += 1;
    }
}

/// Primitive that literally inserts a fence after an `Inst`.
fn insert_fence_after_inst(func: &mut Function, inst: Inst, stats: &mut BladeStats, verbose: bool) {
    if func.post_lfence[inst] {
        // do nothing, already had a fence here
    } else {
        if verbose {
            println!("inserting fence after instruction {:?}", func.dfg[inst]);
        }
        func.post_lfence[inst] = true;
        stats.static_fences_inserted += 1;
    }
}

/// Get the first instruction of the linear block containing the given `Inst`.
/// Linear blocks are not to be confused with basic blocks, nor with the _EBB_ or
/// "extended basic block" (which is what Cranelift considers a "block").
/// For our purposes in this function, all branch, call, and ret instructions
/// terminate blocks. In contrast, in Cranelift, only unconditional branch and
/// ret instructions terminate EBBs, while conditional branches and call
/// instructions do not terminate EBBs.
fn get_first_inst_of_linear_block(func: &Function, inst: Inst) -> Inst {
    let ebb = func
        .layout
        .inst_block(inst)
        .expect("Instruction is not in layout");
    let first_inst = func
        .layout
        .first_inst(ebb)
        .expect("EBB has no instructions");
    let mut cur_inst = inst;
    loop {
        if cur_inst == first_inst {
            // got to beginning of EBB: this is also the top of the linear block
            return first_inst;
        }
        // prev_inst will be the inst above cur_inst
        let prev_inst = func
            .layout
            .prev_inst(cur_inst)
            .expect("Ran off the beginning of the EBB");
        let opcode = func.dfg[prev_inst].opcode();
        if opcode.is_call() || opcode.is_branch() || opcode.is_indirect_branch() {
            // found the previous call or branch instruction:
            // cur_inst is the top of the linear block
            return cur_inst;
        }
        cur_inst = prev_inst;
    }
}

/// Inserts a fence at the beginning of the _linear block_ containing the given
/// instruction. See notes on `get_first_inst_of_linear_block()`, describing how
/// we define linear blocks.
fn insert_fence_at_beginning_of_block(func: &mut Function, inst: Inst, stats: &mut BladeStats, verbose: bool) {
    let top_linear_block = get_first_inst_of_linear_block(func, inst);
    insert_fence_before_inst(func, top_linear_block, stats, verbose);
}

/// Is there a fence at the beginning of the linear block containing the given
/// instruction? See notes on `get_first_inst_of_linear_block()`, describing how
/// we define linear blocks.
fn is_there_a_fence_at_top_of_this_linear_block(func: &Function, inst: Inst) -> bool {
    let top_linear_block = get_first_inst_of_linear_block(func, inst);
    func.pre_lfence[top_linear_block]
}

struct SLHInserter<'a> {
    /// The function we're inserting SLHs into
    func: &'a mut Function,
    /// The `TargetIsa`
    isa: &'a dyn TargetIsa,
    /// tracks which `BladeNode`s have already had SLH applied to them
    bladenodes_done: HashSet<BladeNode>,
    /// the `BladeStats` where we record how many SLHs we're inserting
    stats: &'a mut BladeStats,
    /// if `true`, then we print (to stdout) the location of each SLH as we insert it
    verbose: bool,
}

impl<'a> SLHInserter<'a> {
    /// A blank SLHInserter
    fn new(func: &'a mut Function, isa: &'a dyn TargetIsa, stats: &'a mut BladeStats) -> Self {
        Self {
            func,
            isa,
            bladenodes_done: HashSet::new(),
            stats,
            verbose: DEBUG_PRINT_DETAILED_DEF_LOCATIONS || std::env::var("PRINT_BLADE_DEF_LOCATIONS").is_ok(),
        }
    }

    /// Apply SLH to `bnode`, but only if we haven't already applied SLH to `bnode`
    fn apply_slh_to_bnode(&mut self, bnode: &BladeNode) {
        if self.bladenodes_done.insert(bnode.clone()) {
            _apply_slh_to_bnode(self.func, self.isa, bnode, self.verbose);
            self.stats.static_slhs_inserted += 1;
        }
    }
}

fn _apply_slh_to_bnode(func: &mut Function, isa: &dyn TargetIsa, bnode: &BladeNode, verbose: bool) {
    match bnode {
        BladeNode::Sink(_) => panic!("Can't do SLH to protect a sink, have to protect a source"),
        BladeNode::ValueDef(value) => {
            // The value that needs protecting is `value`, so we need to apply SLH to the load which produced `value`
            match func.dfg.value_def(*value) {
                ValueDef::Param(_, _) => unimplemented!("SLH on a block parameter"),
                ValueDef::Result(inst, _) => {
                    apply_slh_to_load_inst(func, isa, inst, verbose);
                }
            }
        }
    }
}

/// Expects `inst` to be a load instruction, and panics if it is not
fn apply_slh_to_load_inst(func: &mut Function, isa: &dyn TargetIsa, inst: Inst, verbose: bool) {
    assert!(func.dfg[inst].opcode().can_load(), "SLH on a non-load instruction: {:?}", func.dfg[inst]);
    if verbose {
        println!("applying SLH to this load: {:?}", func.dfg[inst]);
    }
    let mut cur = EncCursor::new(func, isa).at_inst(inst);
    // Find the arguments to `inst` which are marked as pointers / have bounds
    // (as pairs (argnum, argvalue))
    let mut pointer_args = cur.func.dfg.inst_args(inst).iter().copied().enumerate().filter(|&(_, arg)| cur.func.dfg.bounds[arg].is_some());
    let (pointer_arg_num, pointer_arg, bounds) = match pointer_args.next() {
        Some((num, arg)) => match pointer_args.next() {
            Some(_) => panic!("SLH: multiple pointer args found to instruction {:?}", func.dfg[inst]),
            None => {
                // all good, there is exactly one pointer arg
                let bounds = cur.func.dfg.bounds[arg].clone().expect("we already checked that there's bounds here");
                (num, arg, bounds)
            }
        }
        None => {
            if ALLOW_FAKE_SLH_BOUNDS {
                let pointer_arg_num = 0; // we pick the first arg, arbitrarily
                let pointer_arg = cur.func.dfg.inst_args(inst)[pointer_arg_num];
                let lower = pointer_arg;
                let upper = cur.ins().iadd_imm(pointer_arg, (FAKE_SLH_ARRAY_LENGTH_BYTES as u64) as i64);
                let bounds = Bounds {
                    lower,
                    upper,
                    directly_annotated: false,
                };
                (pointer_arg_num, pointer_arg, bounds)
            } else {
                panic!("SLH: no pointer arg found for instruction {:?}", func.dfg[inst])
            }
        }
    };
    let masked_pointer = {
        let pointer_ty = cur.func.dfg.value_type(pointer_arg);
        let zero = cur.ins().iconst(pointer_ty, 0);
        let all_ones = cur.ins().iconst(pointer_ty, -1);
        let flags = cur.ins().ifcmp(pointer_arg, bounds.lower);
        let mask = cur.ins().selectif(pointer_ty, IntCC::UnsignedGreaterThanOrEqual, flags, all_ones, zero);
        let op_size_bytes = {
            let results = cur.func.dfg.inst_results(inst);
            assert_eq!(results.len(), 1, "expected load instruction to have exactly one result, got {}", results.len());
            let result = results[0];
            let bytes = cur.func.dfg.value_type(result).bytes() as u64;
            cur.ins().iconst(pointer_ty, bytes as i64)
        };
        let adjusted_upper_bound = cur.ins().isub(bounds.upper, op_size_bytes);
        let flags = cur.ins().ifcmp(pointer_arg, adjusted_upper_bound);
        let mask = cur.ins().selectif(pointer_ty, IntCC::UnsignedLessThanOrEqual, flags, mask, zero);
        cur.ins().band(pointer_arg, mask)
    };
    // now update the original load instruction to use the masked pointer instead
    cur.func.dfg.inst_args_mut(inst)[pointer_arg_num] = masked_pointer;
}

struct DefUseGraph {
    /// Maps a value to its uses
    map: SecondaryMap<Value, Vec<ValueUse>>,
    /// Inverse map: map a value to the values it uses
    inverse_map: SecondaryMap<Value, Vec<Value>>,
}

impl DefUseGraph {
    /// Create a `DefUseGraph` for the given `Function`.
    ///
    /// `cfg`: the `ControlFlowGraph` for the `Function`.
    ///
    /// `call_edges`: if `true`, then for call instructions, the result of the
    /// instruction (i.e., the return value) is considered to be a use of each of
    /// the call arguments. if `false`, then that result (the return value) is
    /// not considered to be a use of anything - it will have no dependents.
    /// This is specifically for call instructions, and does not affect the
    /// return value of this function itself, which will still of course be
    /// considered as using whatever it uses within this function.
    pub fn for_function(func: &Function, cfg: &ControlFlowGraph, call_edges: bool) -> Self {
        let mut map: SecondaryMap<Value, Vec<ValueUse>> =
            SecondaryMap::with_capacity(func.dfg.num_values());
        let mut inverse_map: SecondaryMap<Value, Vec<Value>> =
            SecondaryMap::with_capacity(func.dfg.num_values());

        for block in func.layout.blocks() {
            // Iterate over every instruction. Mark that instruction as a use of
            // each of its parameters.
            // And mark the parameters as dependents of the instruction result(s).
            for inst in func.layout.block_insts(block) {
                // Depending on the `call_edges` option, don't mark the result
                // of a call as a use of each of its parameters
                if !call_edges && func.dfg[inst].opcode().is_call() {
                    continue;
                }
                let results = func.dfg.inst_results(inst);
                for arg in func.dfg.inst_args(inst) {
                    map[*arg].push(ValueUse::Inst(inst));
                    for result in results {
                        inverse_map[*result].push(*arg);
                    }
                }
            }
            // Also, mark each block parameter as a use of the corresponding argument
            // in all branch instructions which can feed this block
            for incoming_bb in cfg.pred_iter(block) {
                let incoming_branch = &func.dfg[incoming_bb.inst];
                let branch_args = match incoming_branch {
                    InstructionData::Branch { .. }
                    | InstructionData::BranchFloat { .. }
                    | InstructionData::BranchIcmp { .. }
                    | InstructionData::BranchInt { .. }
                    | InstructionData::Call { .. }
                    | InstructionData::CallIndirect { .. }
                    | InstructionData::IndirectJump { .. }
                    | InstructionData::Jump { .. } => func.dfg.inst_variable_args(incoming_bb.inst),
                    _ => panic!(
                        "incoming_branch is an unexpected type: {:?}",
                        incoming_branch
                    ),
                };
                let block_params = func.dfg.block_params(block);
                assert_eq!(branch_args.len(), block_params.len());
                for (param, arg) in block_params.iter().zip(branch_args.iter()) {
                    map[*arg].push(ValueUse::Value(*param));
                    inverse_map[*param].push(*arg);
                }
            }
        }

        Self {
            map,
            inverse_map,
        }
    }

    /// Iterate over all the uses of the given `Value`
    pub fn uses_of_val(&self, val: Value) -> impl Iterator<Item = &ValueUse> {
        self.map[val].iter()
    }

    /// Iterate over all the uses of the result of the given `Inst` in the given `Function`
    // (function is currently unused)
    pub fn _uses_of_inst<'a>(
        &'a self,
        inst: Inst,
        func: &'a Function,
    ) -> impl Iterator<Item = &'a ValueUse> {
        func.dfg
            .inst_results(inst)
            .iter()
            .map(move |&val| self.uses_of_val(val))
            .flatten()
    }

    /// Iterate over all the values which a given `Value` uses
    pub fn dependents_of_val<'s>(&'s self, val: Value) -> impl Iterator<Item = Value> + 's {
        self.inverse_map[val].iter().copied()
    }
}

/// Describes a way in which a given `Value` is used
#[derive(Clone, Debug)]
enum ValueUse {
    /// This `Instruction` uses the `Value`
    Inst(Inst),
    /// The `Value` may be forwarded to this `Value`
    Value(Value),
}

struct BladeGraph {
    /// the actual graph
    graph: LinkedListGraph<usize>,
    /// the (single) source node. there are edges from this to sources
    source_node: Node<usize>,
    /// the (single) sink node. there are edges from sinks to this
    sink_node: Node<usize>,
    /// maps graph nodes to the `BladeNode`s which they correspond to
    node_to_bladenode_map: HashMap<Node<usize>, BladeNode>,
    /// maps `BladeNode`s to the graph nodes which they correspond to
    _bladenode_to_node_map: HashMap<BladeNode, Node<usize>>,
}

#[derive(PartialEq, Eq, Clone, Debug, Hash)]
enum BladeNode {
    /// A `BladeNode` representing the definition of a value
    ValueDef(Value),
    /// A `BladeNode` representing an instruction that serves as a sink
    Sink(Inst),
}

impl BladeNode {
    fn unwrap_value(&self) -> Value {
        match self {
            Self::ValueDef(v) => *v,
            Self::Sink(_) => panic!("unwrap_value: not a value"),
        }
    }
}

impl BladeGraph {
    /// Return the cut-edges in the mincut of the graph
    fn min_cut(&self) -> Vec<Edge<usize>> {
        // TODO: our options are `Dinic`, `EdmondsKarp`, or `PushRelabel`.
        // I'm not sure what the tradeoffs are.
        // SC: from my limited wikipedia'ing, pushrelabel is supposedly the best
        let mut maxflow = PushRelabel::<LinkedListGraph<usize>, usize>::new(&self.graph);
        maxflow.solve(self.source_node, self.sink_node, |_| 1); // all edges have weight 1

        // turns out `mincut` returns the set of nodes reachable from the source node after
        //   the graph is cut; we have to recreate the cut based on this set
        let reachable_from_source = maxflow.mincut();
        // XXX there's probably a more efficient algorithm
        reachable_from_source
            .iter()
            .map(move |node| self.graph.outedges(*node))
            .flatten()
            .filter(|(_, dst)| !reachable_from_source.contains(dst))
            .map(|(edge, _)| edge)
            .collect()
    }

    /// Given a `Node`, get a `HashSet` of all of the "source nodes" which have
    /// paths to it, where "source node" is defined by `self.is_source_node()`.
    ///
    /// If the given `node` is itself a "source node", we'll return a set
    /// containing just `node` itself
    fn ancestors_of(&self, node: Node<usize>) -> HashSet<Node<usize>> {
        self._ancestors_of(node, &mut HashSet::new())
    }

    fn _ancestors_of(&self, node: Node<usize>, seen_nodes: &mut HashSet<Node<usize>>) -> HashSet<Node<usize>> {
        if !seen_nodes.insert(node) {
            // we've already processed this node and all its ancestors
            return HashSet::new();
        }
        if self.is_source_node(node) {
            HashSet::from_iter(std::iter::once(node))
        } else {
            let mut hs = HashSet::new();
            for (_, incoming) in self.graph.inedges(node) {
                hs.extend(self._ancestors_of(incoming, seen_nodes))
            }
            hs
        }
    }

    /// Is the given `Node` a "source node" in the graph, where "source node" is
    /// here defined as any node which has an edge from the global source node to
    /// it
    fn is_source_node(&self, node: Node<usize>) -> bool {
        self.graph.inedges(node).any(|(_, incoming)| incoming == self.source_node)
    }

    /// Is the given `Node` a "sink node" in the graph, where "sink node" is
    /// here defined as any node which has an edge to the global sink node
    fn is_sink_node(&self, node: Node<usize>) -> bool {
        self.graph.outedges(node).any(|(_, outgoing)| outgoing == self.sink_node)
    }

    /// Iterate over the "source nodes" in the graph (for the `is_source_node`
    /// sense of "source node")
    fn source_nodes<'s>(&'s self) -> impl Iterator<Item = Node<usize>> + 's {
        self.graph.nodes().filter(move |&node| self.is_source_node(node))
    }

    /// `source_nodes()`, but returns `Value`s instead of `Node`s
    fn source_values<'s>(&'s self) -> impl Iterator<Item = Value> + 's {
        self.graph.nodes()
            .filter(move |&node| self.is_source_node(node))
            .map(move |node| self.node_to_bladenode_map[&node].unwrap_value())  // unwrap_value() is OK because BladeNode::Sink can't be a source
    }

    /// Iterate over the "sink nodes" in the graph (for the `is_sink_node`
    /// sense of "sink node")
    fn sink_nodes<'s>(&'s self) -> impl Iterator<Item = Node<usize>> + 's {
        self.graph.nodes().filter(move |&node| self.is_sink_node(node))
    }

    /// `sink_nodes()`, but returns `BladeNode`s instead of `Node`s
    fn sink_bladenodes<'s>(&'s self) -> impl Iterator<Item = &'s BladeNode> + 's {
        self.graph.nodes()
            .filter(move |&node| self.is_sink_node(node))
            .map(move |node| &self.node_to_bladenode_map[&node])
    }
}

/// `SimpleCache` borrowed from the implementation in the crate
/// [`llvm-ir-analysis`](https://crates.io/crates/llvm-ir-analysis)
struct SimpleCache<T> {
    /// `None` if not computed yet
    data: RefCell<Option<T>>,
}

impl<T> SimpleCache<T> {
    fn new() -> Self {
        Self {
            data: RefCell::new(None),
        }
    }

    /// Get the cached value, or if no value is cached, compute the value using
    /// the given closure, then cache that result and return it
    fn get_or_insert_with(&self, f: impl FnOnce() -> T) -> Ref<T> {
        // borrow mutably only if it's empty. else don't even try to borrow mutably
        let need_mutable_borrow = self.data.borrow().is_none();
        if need_mutable_borrow {
            let old_val = self.data.borrow_mut().replace(f());
            debug_assert!(old_val.is_none());
        }
        // now, either way, it's populated, so we borrow immutably and return.
        // future users can also borrow immutably using this function (even
        // while this borrow is still outstanding), since it won't try to borrow
        // mutably in the future.
        Ref::map(self.data.borrow(), |o| {
            o.as_ref().expect("should be populated now")
        })
    }
}

/// Stores data about which values are BC-tainted
struct BCData {
    /// These values are BC-tainted
    tainted_values: EntitySet<Value>,
}

impl BCData {
    /// Determine which values in the given function are BC-tainted
    fn new(
        func: &Function,
        blade_type: BladeType,
        switchblade_callconv: SwitchbladeCallconv,
        def_use_graph_no_call_edges: &DefUseGraph,
        def_use_graph_with_call_edges: &DefUseGraph,
    ) -> Self {
        // first we collect a list of all the values used as branch conditions
        let branch_conditions = {
            let mut branch_conditions = vec![];
            for block in func.layout.blocks() {
                for inst in func.layout.block_insts(block) {
                    if func.dfg[inst].opcode().is_branch() {
                        // is_branch() covers both conditional branches and indirect branches.
                        // I'm not sure whether we want this or not, but for now we're conservative

                        // `inst_fixed_args` gets the condition args for branches,
                        //   and ignores the destination block params (which are also included in args)
                        branch_conditions.extend(func.dfg.inst_fixed_args(inst));
                    }
                }
            }
            branch_conditions
        };

        // now we find the `roots`: values which influence the branch conditions.
        // Branch conditions are usually conditional expressions like `i > 3`. In
        // this example, `i` is a root; that's what we're doing right now.
        // (We need to trace backwards transitively in order to mark `i` a root in
        // more complicated branch conditions like `(i + 3) / 7 > 56`.)
        let roots = {
            let mut roots = EntitySet::with_capacity(func.dfg.num_values());
            let mut worklist = branch_conditions;  // we start by including all of the branch conditions themselves
            loop {
                match worklist.pop() {
                    None => break,
                    Some(val) => {
                        // if the item is already marked as a root, then we've already processed it and the values which contribute to it
                        if roots.contains(val) {
                            // do nothing
                        } else {
                            // mark the item as a root, which will also prevent us from processing this item again
                            roots.insert(val);
                            // add all (non-constant) values which contribute to it to the worklist.
                            //
                            // This step needs to use the `def_use_graph_with_call_edges`, because
                            // of examples such as Kocher 13, in which `x` needs to be marked BC
                            // during this step.
                            let contributing_vals = def_use_graph_with_call_edges
                                .dependents_of_val(val)
                                .filter(|&v| !value_is_constant(func, v));
                            for v in contributing_vals {
                                worklist.push(v);
                            }
                        }
                    }
                }
            }
            if is_switchblade(&blade_type) {
                // If needed, based on the calling convention, we mark function
                // parameters and/or results of call instructions as BC roots.
                match switchblade_callconv {
                    SwitchbladeCallconv::NotMay | SwitchbladeCallconv::MayMay => {
                        // Since these calling conventions allow return values
                        // to be BC, callees are not required to ensure that
                        // their return values are not BC, and thus callers
                        // (that's us) must pessimistically assume that return
                        // values are BC.
                        // We mark all results of call instructions as BC roots.
                        // We don't need to backtrace from them (which is why we
                        // didn't add them to the worklist above), but we do
                        // need to forward- trace from them (which is why we add
                        // them to the roots, and not just taint them all at the
                        // end).
                        for block in func.layout.blocks() {
                            for inst in func.layout.block_insts(block) {
                                if func.dfg[inst].opcode().is_call() {
                                    for result in func.dfg.inst_results(inst) {
                                        roots.insert(*result);
                                    }
                                }
                            }
                        }
                    }
                    SwitchbladeCallconv::NotNot | SwitchbladeCallconv::MayNot => (),
                }
                match switchblade_callconv {
                    SwitchbladeCallconv::MayNot | SwitchbladeCallconv::MayMay => {
                        // Since these calling conventions allow call arguments
                        // to be BC, callers are not required to ensure that
                        // the arguments they're passing are not BC, and thus
                        // callees (that's us) must pessimistically assume that
                        // function parameters are BC.
                        // We mark all function parameters as BC roots.
                        // We don't need to backtrace from them (which is why we
                        // didn't add them to the worklist above), but we do
                        // need to forward- trace from them (which is why we add
                        // them to the roots, and not just taint them all at the
                        // end).
                        let entry_block = func
                            .layout
                            .entry_block()
                            .expect("Failed to find entry block");
                        for func_param in func.dfg.block_params(entry_block).iter().skip(1) {
                            // parameters of the entry block == parameters of the function
                            // the skip(1) is because the first param is what
                            // Cranelift uses to supply the linear memory base
                            // address, it's not actually a Wasm function
                            // parameter
                            roots.insert(*func_param);
                        }
                    }
                    SwitchbladeCallconv::NotNot | SwitchbladeCallconv::NotMay => (),
                }
            }
            roots
        };

        // finally, BC-tainted values are defined as the roots and any values
        // which use the roots, including transitively.
        let mut tainted_values = EntitySet::with_capacity(func.dfg.num_values());
        let mut worklist = roots;
        loop {
            match worklist.pop() {
                None => break,
                Some(val) => {
                    // if the value is already tainted, then we've already processed it and its uses
                    if tainted_values.contains(val) {
                        // do nothing
                    } else {
                        // mark the value as tainted, which will also prevent us from processing this value again
                        tainted_values.insert(val);
                        // add all uses of the value to the worklist.
                        //
                        // This step uses the `def_use_graph_no_call_edges`, which means that
                        // even if a call argument is marked BC in this step (or is a root),
                        // the result of that call will not be marked BC (unless it is also a
                        // root, of course).
                        // This is ok because:
                        //   - for calling conventions where return values must not be BC: the
                        //     callee is responsible for fencing if necessary, so it's ok for
                        //     the caller (us) to assume that they are not BC
                        //   - for calling conventions where return values may be BC: we
                        //     already (pessimistically) marked all return values as BC roots
                        //     above, so there's no need to propagate the taint to them now.
                        for v in def_use_graph_no_call_edges.uses_of_val(val) {
                            match *v {
                                ValueUse::Inst(inst_use) => {
                                    // add all the results of `inst` to the worklist
                                    for &result in func.dfg.inst_results(inst_use) {
                                        worklist.insert(result);
                                    }
                                }
                                ValueUse::Value(val_use) => {
                                    worklist.insert(val_use);
                                }
                            }
                        }
                    }
                }
            }
        }

        Self {
            tainted_values,
        }
    }
}

struct BladeGraphBuilder {
    /// builder for the actual graph
    graph: <LinkedListGraph<usize> as rs_graph::Buildable>::Builder,
    /// the (single) source node
    source_node: Node<usize>,
    /// the (single) sink node
    sink_node: Node<usize>,
    /// maps graph nodes to the `BladeNode`s which they correspond to
    node_to_bladenode_map: HashMap<Node<usize>, BladeNode>,
    /// maps `BladeNode`s to the graph nodes which they correspond to
    bladenode_to_node_map: HashMap<BladeNode, Node<usize>>,
}

impl BladeGraphBuilder {
    /// Creates a new `BladeGraphBuilder` with the `node_to_bladenode_map` and
    /// `bladenode_to_node_map` populated for all `Value`s in the `Function`
    fn with_nodes_for_func(func: &Function) -> Self {
        let mut gg = LinkedListGraph::<usize>::new_builder();
        let mut node_to_bladenode_map = HashMap::new();
        let mut bladenode_to_node_map = HashMap::new();
        let source_node = gg.add_node();
        let sink_node = gg.add_node();

        // add nodes for all values in the function, and populate our maps accordingly
        for val in func.dfg.values() {
            let node = gg.add_node();
            node_to_bladenode_map.insert(node, BladeNode::ValueDef(val));
            bladenode_to_node_map.insert(BladeNode::ValueDef(val), node);
        }

        Self {
            graph: gg,
            source_node,
            sink_node,
            node_to_bladenode_map,
            bladenode_to_node_map,
        }
    }

    /// Mark the given `Value` as a source.
    fn mark_as_source(&mut self, src: Value) {
        let node = self.bladenode_to_node_map[&BladeNode::ValueDef(src)];
        self.graph.add_edge(self.source_node, node);
    }

    /// Add an edge from the given `Node` to the given `Value`
    fn add_edge_from_node_to_value(&mut self, from: Node<usize>, to: Value) {
        let value_node = self.bladenode_to_node_map[&BladeNode::ValueDef(to)];
        self.graph.add_edge(from, value_node);
    }

    /// Add an edge from the given `Value` to the given `Node`
    fn add_edge_from_value_to_node(&mut self, from: Value, to: Node<usize>) {
        let value_node = self.bladenode_to_node_map[&BladeNode::ValueDef(from)];
        self.graph.add_edge(value_node, to);
    }

    /// Add a new sink node for the given `inst`
    fn add_sink_node_for_inst(&mut self, inst: Inst) -> Node<usize> {
        let inst_sink_node = self.graph.add_node();
        self.node_to_bladenode_map
            .insert(inst_sink_node, BladeNode::Sink(inst));
        self.bladenode_to_node_map
            .insert(BladeNode::Sink(inst), inst_sink_node);
        self.graph.add_edge(inst_sink_node, self.sink_node);
        inst_sink_node
    }

    /// Consumes the `BladeGraphBuilder`, generating a `BladeGraph`
    fn build(self) -> BladeGraph {
        BladeGraph {
            graph: self.graph.to_graph(),
            source_node: self.source_node,
            sink_node: self.sink_node,
            node_to_bladenode_map: self.node_to_bladenode_map,
            _bladenode_to_node_map: self.bladenode_to_node_map,
        }
    }
}

/// Which loads should be considered sources?
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sources {
    /// All loads, including loads with constant addresses
    AllLoads,
    /// Only loads with non-constant addresses
    NonConstantAddr,
    /// Only loads with BC addresses. (No constants are BC, so this will be a subset of the loads with `NonConstantAddr`.)
    BCAddr,
}

/// Is the given `inst` (representing a load instruction) a constant-addr load
/// instruction?
fn load_is_constant_addr(func: &Function, inst: Inst) -> bool {
    let _idata = &func.dfg[inst];
    let _args: Vec<_> = func.dfg.inst_args(inst).iter().collect();
    match &func.dfg[inst] {
        InstructionData::Load { arg, .. } => {
            // the `offset` is always constant, so just check the `arg`
            value_is_constant(func, *arg)
        }
        InstructionData::LoadComplex { args, .. } => {
            // the `offset` is always constant, so just check the args
            args.as_slice(&func.dfg.value_lists).iter().all(|&arg| value_is_constant(func, arg))
        }
        InstructionData::BranchTableEntry { .. } => false, // conservatively
        idata => unimplemented!("load_is_constant_addr: instruction data {:?}", idata),
    }
}

/// Are all the arguments of the given `inst` constant?
fn all_args_are_constant(func: &Function, inst: Inst) -> bool {
    func.dfg.inst_args(inst)
        .iter()
        .all(|&value| value_is_constant(func, value))
}

/// Is the given `Value` a constant?
fn value_is_constant(func: &Function, value: Value) -> bool {
    if let Some(vmctx) = func.special_param(ArgumentPurpose::VMContext) {
        if value == vmctx {
            // the heap base pointer, or other VMContext pointer, is effectively a constant,
            // even though its value may not be known at this stage in the pipeline
            return true;
        }
    }
    match func.dfg.value_def(value) {
        ValueDef::Param(_block, _i) => {
            // conservatively assume that block parameters are not constant
            false
        }
        ValueDef::Result(inst, _) => {
            let opcode = func.dfg[inst].opcode();
            match opcode {
                // constants
                Opcode::Iconst
                | Opcode::F32const
                | Opcode::F64const
                | Opcode::Bconst
                | Opcode::Vconst
                | Opcode::ConstAddr
                | Opcode::JumpTableBase
                | Opcode::Null
                => true,
                // addresses
                Opcode::FuncAddr
                | Opcode::StackAddr
                | Opcode::HeapAddr
                | Opcode::TableAddr
                // arithmetic/bitwise ops etc
                | Opcode::Iadd
                | Opcode::IaddImm
                | Opcode::UaddSat
                | Opcode::SaddSat
                | Opcode::Isub
                | Opcode::IrsubImm
                | Opcode::UsubSat
                | Opcode::SsubSat
                | Opcode::Ineg
                | Opcode::Imul
                | Opcode::ImulImm
                | Opcode::Umulhi
                | Opcode::X86Umulx
                | Opcode::Smulhi
                | Opcode::X86Smulx
                | Opcode::Udiv
                | Opcode::UdivImm
                | Opcode::X86Udivmodx
                | Opcode::Sdiv
                | Opcode::SdivImm
                | Opcode::X86Sdivmodx
                | Opcode::Urem
                | Opcode::UremImm
                | Opcode::Srem
                | Opcode::SremImm
                | Opcode::Imin
                | Opcode::Umin
                | Opcode::Imax
                | Opcode::Umax
                | Opcode::Band
                | Opcode::BandImm
                | Opcode::BandNot
                | Opcode::Bor
                | Opcode::BorImm
                | Opcode::BorNot
                | Opcode::Bxor
                | Opcode::BxorImm
                | Opcode::BxorNot
                | Opcode::Bnot
                | Opcode::Rotl
                | Opcode::RotlImm
                | Opcode::Rotr
                | Opcode::RotrImm
                | Opcode::Ishl
                | Opcode::IshlImm
                | Opcode::Ushr
                | Opcode::UshrImm
                | Opcode::Sshr
                | Opcode::SshrImm
                | Opcode::Bitrev
                | Opcode::IaddCin
                | Opcode::IaddIfcin
                | Opcode::IaddCout
                | Opcode::IaddIfcout
                | Opcode::IaddCarry
                | Opcode::IaddIfcarry
                | Opcode::IsubBin
                | Opcode::IsubIfbin
                | Opcode::IsubBout
                | Opcode::IsubIfbout
                | Opcode::IsubBorrow
                | Opcode::IsubIfborrow
                | Opcode::Select
                | Opcode::Selectif
                | Opcode::Bitselect
                // comparisons
                | Opcode::Icmp
                | Opcode::IcmpImm
                | Opcode::Ifcmp
                | Opcode::IfcmpImm
                | Opcode::Fcmp
                | Opcode::Ffcmp
                // FP ops
                | Opcode::Fadd
                | Opcode::Fsub
                | Opcode::Fmul
                | Opcode::Fdiv
                | Opcode::Sqrt
                | Opcode::Fma
                | Opcode::Fneg
                | Opcode::Fabs
                | Opcode::Fcopysign
                | Opcode::Fmin
                | Opcode::X86Fmin
                | Opcode::Fmax
                | Opcode::X86Fmax
                | Opcode::Ceil
                | Opcode::Floor
                | Opcode::Trunc
                | Opcode::Nearest
                | Opcode::FcvtToUint
                | Opcode::FcvtToUintSat
                | Opcode::FcvtToSint
                | Opcode::FcvtToSintSat
                | Opcode::FcvtFromUint
                | Opcode::FcvtFromSint
                // vector ops
                | Opcode::Insertlane
                | Opcode::Extractlane
                | Opcode::Splat
                | Opcode::Swizzle
                | Opcode::Vsplit
                | Opcode::Vconcat
                | Opcode::Vselect
                | Opcode::VanyTrue
                | Opcode::VallTrue
                | Opcode::X86Pshufb
                | Opcode::X86Pshufd
                | Opcode::X86Pextr
                | Opcode::X86Pinsr
                | Opcode::X86Insertps
                | Opcode::X86Movsd
                | Opcode::X86Movlhps
                | Opcode::X86Psll
                | Opcode::X86Psrl
                | Opcode::X86Psra
                | Opcode::X86Ptest
                | Opcode::X86Pmaxs
                | Opcode::X86Pmaxu
                | Opcode::X86Pmins
                | Opcode::X86Pminu
                // Popcnt etc
                | Opcode::Clz
                | Opcode::Cls
                | Opcode::Ctz
                | Opcode::Popcnt
                // other ops
                | Opcode::IsNull
                | Opcode::IsInvalid
                | Opcode::Trueif
                | Opcode::Trueff
                | Opcode::Breduce
                | Opcode::Bextend
                | Opcode::Bint
                | Opcode::Bmask
                | Opcode::Ireduce
                | Opcode::Uextend
                | Opcode::Sextend
                | Opcode::Fpromote
                | Opcode::Fdemote
                | Opcode::X86Cvtt2si
                | Opcode::X86Bsf
                | Opcode::X86Bsr
                // copies
                | Opcode::Copy
                | Opcode::CopySpecial
                | Opcode::CopyToSsa
                | Opcode::CopyNop
                | Opcode::Regmove
                | Opcode::Bitcast
                | Opcode::RawBitcast
                | Opcode::ScalarToVector
                | Opcode::Spill  // behaves like a copy, see docs on it
                | Opcode::Nop
                => all_args_are_constant(func, inst),
                Opcode::Load
                | Opcode::LoadComplex
                | Opcode::Uload8
                | Opcode::Uload8Complex
                | Opcode::Uload16
                | Opcode::Uload16Complex
                | Opcode::Uload32
                | Opcode::Uload32Complex
                | Opcode::Sload8
                | Opcode::Sload8Complex
                | Opcode::Sload16
                | Opcode::Sload16Complex
                | Opcode::Sload32
                | Opcode::Sload32Complex
                | Opcode::Uload8x8
                | Opcode::Uload16x4
                | Opcode::Uload32x2
                | Opcode::Sload8x8
                | Opcode::Sload16x4
                | Opcode::Sload32x2
                | Opcode::StackLoad
                | Opcode::Fill
                | Opcode::FillNop
                | Opcode::Regfill
                | Opcode::X86Pop
                | Opcode::GetPinnedReg
                | Opcode::SetPinnedReg
                | Opcode::IfcmpSp
                | Opcode::Call
                | Opcode::CallIndirect
                | Opcode::JumpTableEntry
                | Opcode::GlobalValue  // ?? conservative for now
                | Opcode::SymbolValue  // ?? conservative for now
                | Opcode::TlsValue  // ?? conservative for now
                => false,
                Opcode::Jump
                | Opcode::Fallthrough
                | Opcode::Return
                | Opcode::Brz
                | Opcode::Brnz
                | Opcode::BrIcmp
                | Opcode::Brif
                | Opcode::Brff
                | Opcode::BrTable
                | Opcode::Store
                | Opcode::StoreComplex
                | Opcode::Istore8
                | Opcode::Istore8Complex
                | Opcode::Istore16
                | Opcode::Istore16Complex
                | Opcode::Istore32
                | Opcode::Istore32Complex
                | Opcode::StackStore
                | Opcode::X86Push
                | Opcode::Regspill
                | Opcode::AdjustSpDown
                | Opcode::AdjustSpUpImm
                | Opcode::AdjustSpDownImm
                | Opcode::Trap
                | Opcode::Trapz
                | Opcode::Trapnz
                | Opcode::Trapif
                | Opcode::Trapff
                | Opcode::ResumableTrap
                => panic!("Didn't expect instruction with opcode {:?} to produce a value", opcode),
                _ => unimplemented!("value_is_constant: opcode {:?}", opcode),
            }
        }
    }
}

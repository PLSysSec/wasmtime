//! A pass over Cranelift IR which implements the Blade algorithm

use crate::cursor::{Cursor, EncCursor};
use crate::entity::SecondaryMap;
use crate::flowgraph::ControlFlowGraph;
use crate::ir::{condcodes::IntCC, dfg::Bounds, ArgumentPurpose, Function, Inst, InstBuilder, InstructionData, Opcode, Value, ValueDef};
use crate::isa::TargetIsa;
use crate::settings::BladeType;
use rs_graph::linkedlistgraph::{Edge, LinkedListGraph, Node};
use rs_graph::maxflow::pushrelabel::PushRelabel;
use rs_graph::maxflow::MaxFlow;
use rs_graph::traits::{Directed, GraphSize};
use rs_graph::Buildable;
use rs_graph::Builder;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use alloc::vec::Vec;

/// If this is `true`, then Blade will use a fake bound information for any
/// address which does not have associated bounds information.
pub(crate) const ALLOW_FAKE_SLH_BOUNDS: bool = true;

/// Nonsense array length (in bytes) to use as the fake bound information (see
/// `ALLOW_FAKE_SLH_BOUNDS` above)
pub(crate) const FAKE_SLH_ARRAY_LENGTH_BYTES: u32 = 2345;

const DEBUG_PRINT_FUNCTION_BEFORE_AND_AFTER: bool = false;

/// Should we print the (static) count of fences/SLHs
const PRINT_FENCE_COUNTS: bool = true;

struct FenceCounts {
    static_fences_inserted: usize,
    static_slhs_inserted: usize,
}

impl FenceCounts {
    fn new() -> Self {
        Self {
            static_fences_inserted: 0,
            static_slhs_inserted: 0,
        }
    }
}

pub fn do_blade(func: &mut Function, isa: &dyn TargetIsa, cfg: &ControlFlowGraph) {
    if isa.flags().blade_type() == BladeType::None {
        return;
    }

    if DEBUG_PRINT_FUNCTION_BEFORE_AND_AFTER {
        println!("Function before blade:\n{}", func.display(isa));
    }

    let mut fence_counts = FenceCounts::new();

    let store_values_are_sinks =
        if isa.flags().blade_type() == BladeType::Slh && isa.flags().blade_v1_1() {
            // For SLH to protect from v1.1, store values must be marked as sinks
            true
        } else {
            false
        };
    let constant_addr_loads_are_srcs =
        if isa.flags().blade_v1_1() {
            true
        } else {
            false
        };
    let blade_graph = build_blade_graph_for_func(func, cfg, store_values_are_sinks, constant_addr_loads_are_srcs);

    // insert the fences / SLHs
    let mut slh_ctx = SLHContext::new();
    let blade_type = isa.flags().blade_type();
    match blade_type {
        BladeType::Baseline => {
            for source in blade_graph.graph.nodes().filter(|&node| blade_graph.is_source_node(node)) {
                // insert a fence after every source
                insert_fence_after(
                    func,
                    blade_graph.node_to_bladenode_map.get(&source).unwrap().clone(),
                    blade_type,
                    &mut fence_counts,
                )
            }
        }
        BladeType::Lfence | BladeType::LfencePerBlock => {
            for cut_edge in blade_graph.min_cut() {
                let edge_src = blade_graph.graph.src(cut_edge);
                let edge_snk = blade_graph.graph.snk(cut_edge);
                if edge_src == blade_graph.source_node {
                    // source -> n : fence after n
                    insert_fence_after(
                        func,
                        blade_graph
                            .node_to_bladenode_map
                            .get(&edge_snk)
                            .unwrap()
                            .clone(),
                        blade_type,
                        &mut fence_counts,
                    );
                } else if edge_snk == blade_graph.sink_node {
                    // n -> sink : fence before (def of) n
                    insert_fence_before(
                        func,
                        blade_graph
                            .node_to_bladenode_map
                            .get(&edge_src)
                            .unwrap()
                            .clone(),
                        blade_type,
                        &mut fence_counts,
                    );
                } else {
                    // n -> m : fence before m
                    insert_fence_before(
                        func,
                        blade_graph
                            .node_to_bladenode_map
                            .get(&edge_snk)
                            .unwrap()
                            .clone(),
                        blade_type,
                        &mut fence_counts,
                    );
                }
            }
        }
        BladeType::Slh => {
            for cut_edge in blade_graph.min_cut() {
                let edge_src = blade_graph.graph.src(cut_edge);
                let edge_snk = blade_graph.graph.snk(cut_edge);
                if edge_src == blade_graph.source_node {
                    // source -> n : apply SLH to the instruction that produces n
                    slh_ctx.do_slh_on(func, isa, blade_graph.node_to_bladenode_map[&edge_snk].clone(), &mut fence_counts);
                } else if edge_snk == blade_graph.sink_node {
                    // n -> sink : for SLH we can't cut at n (which is a sink instruction), we have
                    // to trace back through the graph and cut at all sources which lead to n
                    for node in blade_graph.ancestors_of(edge_src) {
                        slh_ctx.do_slh_on(func, isa, blade_graph.node_to_bladenode_map[&node].clone(), &mut fence_counts);
                    }
                } else {
                    // n -> m : likewise, apply SLH to all sources which lead to n
                    for node in blade_graph.ancestors_of(edge_src) {
                        slh_ctx.do_slh_on(func, isa, blade_graph.node_to_bladenode_map[&node].clone(), &mut fence_counts);
                    }
                }
            }
        }
        BladeType::None => panic!("Shouldn't reach here with Blade setting None"),
    }

    if DEBUG_PRINT_FUNCTION_BEFORE_AND_AFTER {
        println!("Function after blade:\n{}", func.display(isa));
    }

    if PRINT_FENCE_COUNTS {
        match blade_type {
            BladeType::Lfence | BladeType::LfencePerBlock | BladeType::Baseline => {
                println!("function {}: inserted {} (static) lfences", func.name, fence_counts.static_fences_inserted);
                assert_eq!(fence_counts.static_slhs_inserted, 0);
            }
            BladeType::Slh => {
                println!("function {}: inserted {} (static) SLHs", func.name, fence_counts.static_slhs_inserted);
                assert_eq!(fence_counts.static_fences_inserted, 0);
            }
            BladeType::None => {
                assert_eq!(fence_counts.static_fences_inserted, 0);
                assert_eq!(fence_counts.static_slhs_inserted, 0);
            }
        }
    }
}

fn insert_fence_before(func: &mut Function, bnode: BladeNode, blade_type: BladeType, fence_counts: &mut FenceCounts) {
    match bnode {
        BladeNode::ValueDef(val) => match func.dfg.value_def(val) {
            ValueDef::Result(inst, _) => {
                match blade_type {
                    BladeType::Lfence | BladeType::Baseline => {
                        // cut at this value by putting lfence before `inst`
                        if func.pre_lfence[inst] {
                            // do nothing, already had a fence here
                        } else {
                            func.pre_lfence[inst] = true;
                            fence_counts.static_fences_inserted += 1;
                        }
                    }
                    BladeType::LfencePerBlock => {
                        // just put one fence at the beginning of the block.
                        // this stops speculation due to branch mispredictions.
                        insert_fence_at_beginning_of_block(func, inst, fence_counts);
                    }
                    _ => panic!(
                        "This function didn't expect to be called with blade_type {:?}",
                        blade_type
                    ),
                }
            }
            ValueDef::Param(block, _) => {
                // cut at this value by putting lfence at beginning of
                // the `block`, that is, before the first instruction
                let first_inst = func
                    .layout
                    .first_inst(block)
                    .expect("block has no instructions");
                if func.pre_lfence[first_inst] {
                    // do nothing, already had a fence there
                } else {
                    func.pre_lfence[first_inst] = true;
                    fence_counts.static_fences_inserted += 1;
                }
            }
        },
        BladeNode::Sink(inst) => {
            match blade_type {
                BladeType::Lfence | BladeType::Baseline => {
                    // cut at this instruction by putting lfence before it
                    if func.pre_lfence[inst] {
                        // do nothing, already had a fence here
                    } else {
                        func.pre_lfence[inst] = true;
                        fence_counts.static_fences_inserted += 1;
                    }
                }
                BladeType::LfencePerBlock => {
                    // just put one fence at the beginning of the block.
                    // this stops speculation due to branch mispredictions.
                    insert_fence_at_beginning_of_block(func, inst, fence_counts);
                }
                _ => panic!(
                    "This function didn't expect to be called with blade_type {:?}",
                    blade_type
                ),
            }
        }
    }
}

fn insert_fence_after(func: &mut Function, bnode: BladeNode, blade_type: BladeType, fence_counts: &mut FenceCounts) {
    match bnode {
        BladeNode::ValueDef(val) => match func.dfg.value_def(val) {
            ValueDef::Result(inst, _) => {
                match blade_type {
                    BladeType::Lfence | BladeType::Baseline => {
                        // cut at this value by putting lfence after `inst`
                        if func.post_lfence[inst] {
                            // do nothing, already had a fence here
                        } else {
                            func.post_lfence[inst] = true;
                            fence_counts.static_fences_inserted += 1;
                        }
                    }
                    BladeType::LfencePerBlock => {
                        // just put one fence at the beginning of the block.
                        // this stops speculation due to branch mispredictions.
                        insert_fence_at_beginning_of_block(func, inst, fence_counts);
                    }
                    _ => panic!(
                        "This function didn't expect to be called with blade_type {:?}",
                        blade_type
                    ),
                }
            }
            ValueDef::Param(block, _) => {
                // cut at this value by putting lfence at beginning of
                // the `block`, that is, before the first instruction
                let first_inst = func
                    .layout
                    .first_inst(block)
                    .expect("block has no instructions");
                if func.pre_lfence[first_inst] {
                    // do nothing, already had a fence here
                } else {
                    func.pre_lfence[first_inst] = true;
                    fence_counts.static_fences_inserted += 1;
                }
            }
        },
        BladeNode::Sink(_) => panic!("Fencing after a sink instruction"),
    }
}

// Inserts a fence at the beginning of the _basic block_ containing the given
// instruction. "Basic block" is not to be confused with the _EBB_ or "extended
// basic block" (which is what Cranelift considers a "block").
// For our purposes in this function, all branch, call, and ret instructions
// terminate blocks. In contrast, in Cranelift, only unconditional branch and
// ret instructions terminate EBBs, while conditional branches and call
// instructions do not terminate EBBs.
fn insert_fence_at_beginning_of_block(func: &mut Function, inst: Inst, fence_counts: &mut FenceCounts) {
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
            // got to beginning of EBB: insert at beginning of EBB
            if func.pre_lfence[first_inst] {
                // do nothing, already had a fence here
            } else {
                func.pre_lfence[first_inst] = true;
                fence_counts.static_fences_inserted += 1;
            }
            break;
        }
        cur_inst = func
            .layout
            .prev_inst(cur_inst)
            .expect("Ran off the beginning of the EBB");
        let opcode = func.dfg[cur_inst].opcode();
        if opcode.is_call() || opcode.is_branch() || opcode.is_indirect_branch() {
            // found the previous call or branch instruction:
            // insert after that call or branch instruction
            if func.post_lfence[cur_inst] {
                // do nothing, already had a fence here
            } else {
                func.post_lfence[cur_inst] = true;
                fence_counts.static_fences_inserted += 1;
            }
            break;
        }
    }
}

struct SLHContext {
    /// tracks which `BladeNode`s have already had SLH applied to them
    bladenodes_done: HashSet<BladeNode>,
}

impl SLHContext {
    /// A blank SLHContext
    fn new() -> Self {
        Self {
            bladenodes_done: HashSet::new(),
        }
    }

    /// Do SLH on `bnode`, but only if we haven't already done SLH on `bnode`
    fn do_slh_on(&mut self, func: &mut Function, isa: &dyn TargetIsa, bnode: BladeNode, fence_counts: &mut FenceCounts) {
        if self.bladenodes_done.insert(bnode.clone()) {
            _do_slh_on(func, isa, bnode);
            fence_counts.static_slhs_inserted += 1;
        }
    }
}

fn _do_slh_on(func: &mut Function, isa: &dyn TargetIsa, bnode: BladeNode) {
    match bnode {
        BladeNode::Sink(_) => panic!("Can't do SLH to protect a sink, have to protect a source"),
        BladeNode::ValueDef(value) => {
            // The value that needs protecting is `value`, so we need to apply SLH to the load which produced `value`
            match func.dfg.value_def(value) {
                ValueDef::Param(_, _) => unimplemented!("SLH on a block parameter"),
                ValueDef::Result(inst, _) => {
                    assert!(func.dfg[inst].opcode().can_load(), "SLH on a non-load instruction: {:?}", func.dfg[inst]);
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
                            let bytes = cur.func.dfg.value_type(value).bytes() as u64;
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
            }
        }
    }
}

struct DefUseGraph {
    /// Maps a value to its uses
    map: SecondaryMap<Value, Vec<ValueUse>>,
}

impl DefUseGraph {
    /// Create a `DefUseGraph` for the given `Function`.
    ///
    /// `cfg`: the `ControlFlowGraph` for the `Function`.
    pub fn for_function(func: &Function, cfg: &ControlFlowGraph) -> Self {
        let mut map: SecondaryMap<Value, Vec<ValueUse>> =
            SecondaryMap::with_capacity(func.dfg.num_values());

        for block in func.layout.blocks() {
            // Iterate over every instruction. Mark that instruction as a use of
            // each of its parameters.
            for inst in func.layout.block_insts(block) {
                for arg in func.dfg.inst_args(inst) {
                    map[*arg].push(ValueUse::Inst(inst));
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
                }
            }
        }

        Self { map }
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

    /// Given a `Node`, iterate over all of the "source nodes" which have paths
    /// to it, where "source node" is defined by `self.is_source_node`.
    ///
    /// If the given `node` is itself a "source node", we'll just return `node` itself
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

/// `store_values_are_sinks`: if `true`, then the value operand to a store
/// instruction is considered a sink. if `false`, it is not.
/// For instance in the instruction "store x to addrA", if
/// `store_values_are_sinks` is `true`, then both `x` and `addrA` are sinks,
/// but if it is `false`, then just `addrA` is a sink.
fn build_blade_graph_for_func(
    func: &mut Function,
    cfg: &ControlFlowGraph,
    store_values_are_sinks: bool,
    constant_addr_loads_are_srcs: bool,
) -> BladeGraph {
    let mut builder = BladeGraphBuilder::with_nodes_for_func(func);

    // find sources and sinks, and add edges to/from our global source and sink nodes
    for block in func.layout.blocks() {
        for inst in func.layout.block_insts(block) {
            let idata = &func.dfg[inst];
            let op = idata.opcode();
            if op.can_load() {
                // loads are both sources (their loaded values) and sinks (their addresses)
                // except for fills, which don't have sinks

                // handle load as a source
                if constant_addr_loads_are_srcs || !load_is_constant_addr(func, inst) {
                    for &result in func.dfg.inst_results(inst) {
                        builder.mark_as_source(result);
                    }
                }

                // handle load as a sink, except for fills
                if !(op == Opcode::Fill || op == Opcode::FillNop) {
                    let inst_sink_node = builder.add_sink_node_for_inst(inst);
                    // for each address component variable of inst,
                    // add edge address_component_variable_node -> sink
                    // XXX X86Pop has an implicit dependency on %rsp which is not captured here
                    for &arg in func.dfg.inst_args(inst) {
                        builder.add_edge_from_value_to_node(arg, inst_sink_node);
                    }
                }

            } else if op.can_store() {
                // loads are both sources and sinks, but stores are just sinks

                let inst_sink_node = builder.add_sink_node_for_inst(inst);
                // similar to the load case above, but special treatment for the value being stored
                // XXX X86Push has an implicit dependency on %rsp which is not captured here
                if store_values_are_sinks {
                    for &arg in func.dfg.inst_args(inst) {
                        builder.add_edge_from_value_to_node(arg, inst_sink_node);
                    }
                } else {
                    // SC: as far as I can tell, all stores (that have arguments) always
                    //   have the value being stored as the first argument
                    //   and everything after is address args
                    for &arg in func.dfg.inst_args(inst).iter().skip(1) { // skip the first argument
                        builder.add_edge_from_value_to_node(arg, inst_sink_node);
                    }
                };

            } else if op.is_branch() {
                // conditional branches are sinks

                let inst_sink_node = builder.add_sink_node_for_inst(inst);

                // blade only does conditional branches but this will handle indirect jumps as well
                // `inst_fixed_args` gets the condition args for branches,
                //   and ignores the destination block params (which are also included in args)
                for &arg in func.dfg.inst_fixed_args(inst) {
                    builder.add_edge_from_value_to_node(arg, inst_sink_node);
                }

            }
            if op.is_call() {
                // call instruction: must assume that the return value(s) could be a source
                for &result in func.dfg.inst_results(inst) {
                    builder.mark_as_source(result);
                }

                // and, to avoid interprocedural analysis, we require that
                // function arguments are stable, so we mark arguments to a call
                // as sinks
                let inst_sink_node = builder.add_sink_node_for_inst(inst);
                for &arg in func.dfg.inst_args(inst) {
                    builder.add_edge_from_value_to_node(arg, inst_sink_node);
                }
            }
        }
    }

    // we no longer mark function parameters as transient, since we require that
    // they are stable on the caller side (so this is commented)
    /*
    let entry_block = func
        .layout
        .entry_block()
        .expect("Failed to find entry block");
    for &func_param in func.dfg.block_params(entry_block) {
        // parameters of the entry block == parameters of the function
        builder.mark_as_source(func_param);
    }
    */

    // now add edges for actual data dependencies
    // for instance in the following pseudocode:
    //     x = load y
    //     z = x + 2
    //     branch on z
    // we have z -> sink and source -> x, but need x -> z yet
    let def_use_graph = DefUseGraph::for_function(func, cfg);
    for val in func.dfg.values() {
        let node = builder.bladenode_to_node_map[&BladeNode::ValueDef(val)]; // must exist
        for val_use in def_use_graph.uses_of_val(val) {
            match *val_use {
                ValueUse::Inst(inst_use) => {
                    // add an edge from val to the result of inst_use
                    // TODO this assumes that all results depend on all operands;
                    // are there any instructions where this is not the case for our purposes?
                    for &result in func.dfg.inst_results(inst_use) {
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

    builder.build()
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
                | Opcode::Smulhi
                | Opcode::Udiv
                | Opcode::UdivImm
                | Opcode::Sdiv
                | Opcode::SdivImm
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
                | Opcode::Fmax
                | Opcode::Ceil
                | Opcode::Floor
                | Opcode::Trunc
                | Opcode::Nearest
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
                // copies
                | Opcode::Copy
                | Opcode::CopySpecial
                | Opcode::CopyToSsa
                | Opcode::CopyNop
                | Opcode::Regmove
                | Opcode::Bitcast
                | Opcode::RawBitcast
                | Opcode::ScalarToVector
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
                | Opcode::GetPinnedReg
                | Opcode::SetPinnedReg
                | Opcode::IfcmpSp
                | Opcode::Call
                | Opcode::CallIndirect
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
                | Opcode::Spill
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

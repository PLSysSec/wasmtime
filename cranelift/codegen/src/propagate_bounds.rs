use crate::cursor::{Cursor, FuncCursor};
use crate::ir::{Bounds, DataFlowGraph, Function, Opcode, Value};
use alloc::vec::Vec;

/// Propagate bounds info so that all pointer-ish values are given valid bounds.
pub fn propagate_bounds(func: &mut Function) {
    // may be inefficient, but our current implementation just clears all derived bounds and then
    // recomputes everything. This seems easier than trying to remove derived bounds only when
    // they no longer apply, and then we have to remove bounds from all dependent Values, etc
    clear_all_derived_bounds(&mut func.dfg);

    // fixpoint algorithm to propagate all bounds info
    let mut go_again = true;
    while go_again {
        go_again = false;
        let mut cur = FuncCursor::new(func);
        while let Some(_block) = cur.next_block() {
            while let Some(inst) = cur.next_inst() {
                let input_args_with_bounds: Vec<Value> = cur.func
                    .dfg
                    .inst_args(inst)
                    .iter()
                    .copied()
                    .filter(|&arg| cur.func.dfg.bounds[arg].is_some())
                    .collect();
                match input_args_with_bounds.len() {
                    0 => continue,
                    1 => {
                        let opcode = cur.func.dfg[inst].opcode();
                        match HowToHandlePointerInput::for_opcode(opcode) {
                            HowToHandlePointerInput::PropagateBounds => {
                                // propagate bounds from input to result
                                let result = cur.func.dfg.inst_results(inst);
                                assert_eq!(result.len(), 1, "Didn't expect multiple results here");
                                let result = result[0]; // there's only one
                                let arg = input_args_with_bounds[0]; // there's only one
                                cur.func.dfg.bounds[result] = cur.func.dfg.bounds[arg].clone();
                            }
                            HowToHandlePointerInput::DontPropagateBounds => {
                                continue
                            }
                            HowToHandlePointerInput::Unexpected => {
                                unimplemented!("don't know what to do about pointer input to {:?}", opcode)
                            }
                        }
                    }
                    other => panic!("Instruction {:?} has {} arguments with bounds specified", cur.func.dfg[inst].opcode(), other),
                }
            }
        }
    }
}

/// Clear all derived bounds info, leaving only the bounds info that
/// originated from direct annotations.
fn clear_all_derived_bounds(dfg: &mut DataFlowGraph) {
    for bounds in dfg.bounds.values_mut() {
        if let Some(Bounds {
            directly_annotated: false,
            ..
        }) = bounds
        {
            *bounds = None;
        }
    }
}

/// What should happen if an instruction has a pointer as input
enum HowToHandlePointerInput {
    /// Used for instructions like Add which should propagate bounds from input to result
    PropagateBounds,
    /// Used for instructions like Load which allow pointer inputs, but input bounds have no effect on the bounds of the result
    DontPropagateBounds,
    /// Used for instructions which we don't expect pointer inputs, or we haven't handled yet
    Unexpected,
}

impl HowToHandlePointerInput {
    /// How should an instruction with the given opcode behave if it has pointer input?
    fn for_opcode(opcode: Opcode) -> Self {
        match opcode {
            Opcode::Iadd
            | Opcode::IaddImm
            | Opcode::IaddCin
            | Opcode::IaddIfcin
            | Opcode::IaddCout
            | Opcode::IaddIfcout
            | Opcode::IaddCarry
            | Opcode::IaddIfcarry
            | Opcode::UaddSat
            | Opcode::SaddSat
            | Opcode::Isub
            | Opcode::IsubBin
            | Opcode::IsubIfbin
            | Opcode::IsubBout
            | Opcode::IsubIfbout
            | Opcode::IsubBorrow
            | Opcode::IsubIfborrow
            | Opcode::UsubSat
            | Opcode::SsubSat
            | Opcode::Select
            | Opcode::Selectif
            | Opcode::Iconst
            | Opcode::ConstAddr
            | Opcode::Null
            => Self::PropagateBounds,

            Opcode::Load
            | Opcode::LoadComplex
            | Opcode::Store
            | Opcode::StoreComplex
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
            | Opcode::Istore8
            | Opcode::Istore8Complex
            | Opcode::Istore16
            | Opcode::Istore16Complex
            | Opcode::Istore32
            | Opcode::Istore32Complex
            | Opcode::Uload8x8
            | Opcode::Sload8x8
            | Opcode::Uload16x4
            | Opcode::Sload16x4
            | Opcode::Uload32x2
            | Opcode::Sload32x2
            | Opcode::StackLoad
            | Opcode::StackStore
            | _ if opcode.is_branch() || opcode.is_indirect_branch()
            => Self::DontPropagateBounds,

            _ => Self::Unexpected,
        }
    }
}

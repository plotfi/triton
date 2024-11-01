/// __FACEBOOK__ (facebook) begin T203329359
/// Allocate shared memory
def TT_LocalCopyOp : TT_Op<"local_copy", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "copy tensor to local memory";
  let arguments = (ins TT_Tensor:$src);

  let extraClassDeclaration = [{
    bool isSharedMemoryAlloc() {
      return true;
    }
  }];
  let assemblyFormat = [{$src attr-dict `:` functional-type(operands, results)}];

  let results = (outs TT_MemDescType:$result);
}

def TT_GatherOp : TT_Op<"gather", [
  AttrSizedOperandSegments,
  MemoryEffects<[MemRead<SharedMemory>]>,
  TypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
  TypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 4) || std::equal_to<>()">
]> {
  let summary = "Gather from a buffer into a distributed tensor";

  let arguments = (
      ins TT_MemDescType:$src,
      TT_IntTensor:$indices,
      Optional<TT_BoolLike>:$mask,
      Optional<TT_Type>:$other
  );

  let assemblyFormat = [{
    $src `[` $indices `]` (`,` $mask^)? (`,` $other^)?
    attr-dict `:`
    `(` type($src) `,` type($indices)
       (`,` type($mask)^ )? (`,` type($other)^ )?
    `)` `->` type($result)
  }];

  let results = (outs TT_Tensor:$result);
}
/// __FACEBOOK__ (facebook) end T203329359

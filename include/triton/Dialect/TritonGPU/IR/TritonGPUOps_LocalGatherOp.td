/// __FACEBOOK__ (facebook) begin T203329359
def TTG_LocalGatherOp : TTG_Op<"local_gather", [
  AttrSizedOperandSegments,
  MemoryEffects<[MemRead<SharedMemory>]>,
  TypesMatchWith<"infer mask type from src type",
                 "src", "mask", "getI1SameShape($_self)",
                 "($_op.getOperands().size() <= 3) || std::equal_to<>()">,
  TypesMatchWith<"infer other type from src type",
                 "src", "other", "getPointeeType($_self)",
                 "($_op.getOperands().size() <= 4) || std::equal_to<>()">
]> {
  let summary = "Gather a buffer from local memory into a distributed tensor";

  let description = [{
    Gather a tensor from the local memory descriptor into a distributed tensor.
  }];
  let arguments = (
      ins TT_MemDescType:$src,
      TT_IntTensor:$indices,
      Optional<TT_BoolLike>:$mask,
      Optional<TT_Type>:$other // TODO: other is not yet used
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

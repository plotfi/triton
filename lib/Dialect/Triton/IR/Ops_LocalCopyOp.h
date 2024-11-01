/// __FACEBOOK__ (facebook) begin T203329359
void LocalCopyOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  if (!getType().getMutableMemory() && !op->hasAttr("allocation.offset"))
    return;
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       mlir::triton::SharedMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(),
                         mlir::triton::SharedMemory::get());
}
/// __FACEBOOK__ (facebook) end T203329359

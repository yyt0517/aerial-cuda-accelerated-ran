// Experiment-only, causal export. Invoked before sending PreActionRsp.
// Does not change simulator state or consume random numbers.
static void writeDirectedSnapshot(
    int tti, const cumac::cumacCellGrpPrms* gpu,
    const cumac::cumacCellGrpUeStatus* cpu) {
  const char* path = std::getenv("CUMAC_DIRECTED_SNAPSHOT_PATH");
  if (!path || !*path) return;
  const uint32_t u = gpu->nActiveUe, c = gpu->totNumCell, p = gpu->nPrbGrp;
  const uint32_t nr = gpu->nUeAnt, nt = gpu->nBsAnt;
  if (u != 36 || c != 3 || p != 17 || nr != 4 || nt != 4) {
    throw std::runtime_error("directed probe only validated for 3cell/36UE/17PRG/4T4R");
  }
  std::vector<cuComplex> h(static_cast<size_t>(p)*u*c*nr*nt);
  std::vector<cuComplex> w(static_cast<size_t>(u)*p*nt*nt);
  CUDA_CHECK_ERR(cudaMemcpy(h.data(), gpu->estH_fr_actUe, h.size()*sizeof(cuComplex), cudaMemcpyDeviceToHost));
  CUDA_CHECK_ERR(cudaMemcpy(w.data(), gpu->prdMat_actUe, w.size()*sizeof(cuComplex), cudaMemcpyDeviceToHost));
  std::vector<float> queue(u);
  for (uint32_t i=0; i<u; ++i) queue[i] = static_cast<float>(cpu->bufferSize[i]);
  const uint32_t header[] = {0x44494331U, static_cast<uint32_t>(tti), u, c, p, nr, nt, 1U};
  const std::string temporary = std::string(path)+".tmp";
  std::ofstream out(temporary, std::ios::binary|std::ios::trunc);
  out.write(reinterpret_cast<const char*>(header), sizeof(header));
  out.write(reinterpret_cast<const char*>(&gpu->sigmaSqrd), sizeof(float));
  out.write(reinterpret_cast<const char*>(queue.data()), queue.size()*sizeof(float));
  out.write(reinterpret_cast<const char*>(h.data()), h.size()*sizeof(cuComplex));
  out.write(reinterpret_cast<const char*>(w.data()), w.size()*sizeof(cuComplex));
  out.close();
  if (!out || std::rename(temporary.c_str(), path) != 0) {
    throw std::runtime_error("failed to write causal directed snapshot");
  }
}

// Experiment-local current observation export, synchronized by PreActionRsp.
// No simulator mutation or RNG calls. The file is complete before the response.
template <typename T>
static void writeFreshVector(std::ofstream& file, const std::vector<T>& values) {
  file.write(reinterpret_cast<const char*>(values.data()), values.size()*sizeof(T));
}

static bool writeFreshObservation(
    int tti, int64_t snapshotId, uint32_t nSchedUe,
    const std::vector<float>& cell, const std::vector<float>& ue,
    const std::vector<float>& prg, const std::vector<int16_t>& edgeIndex,
    const std::vector<float>& edgeAttr, const std::vector<uint8_t>& maskUe,
    const std::vector<uint8_t>& maskCellUe, const std::vector<uint8_t>& maskPrg,
    const std::vector<float>& postEqSinr, const std::vector<int32_t>& teacherUe,
    const std::vector<int16_t>& teacherPrg) {
  const char* path = std::getenv("CUMAC_FRESH_OBSERVATION_PATH");
  if (!path || !*path) return false;
  const uint64_t header[] = {
      0x46524F31ULL, 1ULL, static_cast<uint64_t>(tti),
      static_cast<uint64_t>(snapshotId), nSchedUe,
      cell.size(), ue.size(), prg.size(), edgeIndex.size(), edgeAttr.size(),
      maskUe.size(), maskCellUe.size(), maskPrg.size(), postEqSinr.size(),
      teacherUe.size(), teacherPrg.size()};
  const std::string temporary = std::string(path)+".tmp";
  std::ofstream file(temporary, std::ios::binary|std::ios::trunc);
  file.write(reinterpret_cast<const char*>(header), sizeof(header));
  writeFreshVector(file, cell);
  writeFreshVector(file, ue);
  writeFreshVector(file, prg);
  writeFreshVector(file, edgeIndex);
  writeFreshVector(file, edgeAttr);
  writeFreshVector(file, maskUe);
  writeFreshVector(file, maskCellUe);
  writeFreshVector(file, maskPrg);
  writeFreshVector(file, postEqSinr);
  writeFreshVector(file, teacherUe);
  writeFreshVector(file, teacherPrg);
  file.close();
  if (!file || std::rename(temporary.c_str(), path) != 0) {
    throw std::runtime_error("failed to write fresh pre-action observation");
  }
  return true;
}

"""Extend the frozen three-cell boundary sampler to adjacent hexagonal sites."""
def extend(text):
    old='''    if ((placementCfg.mode == UePlacementMode::CooperativeCenter ||
         placementCfg.mode == UePlacementMode::CooperativeBoundary) &&
        netData->numCoorCell != 3) {'''
    new='''    if ((placementCfg.mode == UePlacementMode::CooperativeCenter && netData->numCoorCell != 3) ||
        (placementCfg.mode == UePlacementMode::CooperativeBoundary &&
         netData->numCoorCell != 3 && netData->numCoorCell != 7)) {'''
    assert text.count(old)==1;text=text.replace(old,new)
    start=text.index('                    int pairOrdinal = uIdx % std::max(1, netData->numCoorCell - 1);')
    stop=text.index('                    const float neighborX',start)
    text=text[:start]+'''                    // Boundary means a shared nearest-neighbor Voronoi edge.
                    // All pairs are adjacent in the original 3-cell triangle.
                    std::vector<int> adjacent;
                    for (int other = 0; other < netData->numCoorCell; ++other) {
                        if (other == cIdx) continue;
                        const float dx = netData->bsPos[other][0] - coorX;
                        const float dy = netData->bsPos[other][1] - coorY;
                        if (std::abs(std::sqrt(dx*dx + dy*dy) - siteSpacing) < 1.e-3f * siteSpacing)
                            adjacent.push_back(other);
                    }
                    if (adjacent.empty()) throw std::runtime_error("No adjacent boundary site");
                    const int neighborCellIdx = adjacent[uIdx % adjacent.size()];

'''+text[stop:]
    return text

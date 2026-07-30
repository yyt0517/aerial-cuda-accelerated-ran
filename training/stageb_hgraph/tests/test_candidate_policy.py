from training.stageb_hgraph.graph_ops import pack_prg_candidates
from training.stageb_hgraph.tests.graph_fixtures import make_candidate_graph


def test_candidate_fixture_has_three_cells_and_all_demand_ues():
    graph = make_candidate_graph(n_prg=2)
    packed = pack_prg_candidates(graph, max_candidates=2, fixed_width=True)
    assert graph.snapshot.static_meta.n_cell == 3
    assert graph.snapshot.static_meta.n_prg == 2
    assert packed.ue_ids.shape == (6, 2)
    assert packed.mask.all()

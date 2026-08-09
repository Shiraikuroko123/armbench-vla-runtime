# CPU runtime completion matrix

Provider-neutral asynchronous assurance boundary; scripted/frozen/contract fixtures only.

- Cases: 17
- Expected outcomes: 17/17
- Complete plans published: 6
- Holds: 10
- Unrecoverable stops: 1
- Partial policy prefixes exposed: 0
- Assurance worker P95: 281.200 ms

| Case | Provider | Fault | Status | Reason | Match |
| --- | --- | --- | --- | --- | ---: |
| mock_latency_000ms | mock_hx8 | none | execute | qp_continuous_collision_and_braking_invariant_passed | True |
| mock_latency_040ms | mock_hx8 | none | execute | qp_continuous_collision_and_braking_invariant_passed | True |
| mock_latency_080ms | mock_hx8 | none | execute | qp_continuous_collision_and_braking_invariant_passed | True |
| mock_latency_160ms | mock_hx8 | none | execute | qp_continuous_collision_and_braking_invariant_passed | True |
| frozen_provider_nominal | frozen_hx7_adapter | none | execute | qp_continuous_collision_and_braking_invariant_passed | True |
| provider_interface_fixture_nominal | openpi_interface_fixture | none | execute | qp_continuous_collision_and_braking_invariant_passed | True |
| provider_fault_malformed_shape | fault_injector | malformed_shape | hold | provider_failure:TypeError | True |
| provider_fault_nonfinite | fault_injector | nonfinite | hold | provider_failure:ValueError | True |
| provider_fault_disconnect | fault_injector | disconnect | hold | provider_failure:ConnectionError | True |
| provider_fault_timeout | fault_injector | timeout | hold | provider_failure:TimeoutError | True |
| provider_fault_sequence_mismatch | fault_injector | sequence_mismatch | hold | provider_failure:ValueError | True |
| stale_observation | mock_hx8 | stale | hold | response_deadline_exceeded_before_supervision | True |
| state_mismatch | mock_hx8 | state_mismatch | hold | observation_state_mismatch | True |
| reset_replay | mock_hx8 | reset_before_commit | hold | reset_generation_mismatch | True |
| supervision_budget_fail_closed | mock_hx8 | tiny_supervision_budget | hold | supervision_budget_exceeded | True |
| near_limit_unrecoverable_stop | mock_hx8 | near_limit_stop | unrecoverable_stop | response_deadline_exceeded_before_supervision | True |
| intermediate_collision_fail_closed | mock_hx8 | intermediate_self_collision | hold | continuous_edge_rejected:collision:self:link0:geom_12:link6:geom_50 | True |

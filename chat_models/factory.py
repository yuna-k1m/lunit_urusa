from __future__ import annotations

from chat_models.direct_l2 import DirectL2Strategy
from chat_models.demographics import DemographicsStrategy
from chat_models.multi_patient import MultiPatientStrategy
from chat_models.patient_sim import PatientSimStrategy
from chat_models.registry import StrategyRegistry
from chat_models.siusiubeom import SiusiubeomH4Strategy
from clients.lunit import LunitClient
from clients.l2_plus import L2PlusClient
from clients.openai_sol import SolClient
from clients.patient_simulator import PatientSimulatorClient
from config import Settings


def build_registry(settings: Settings) -> StrategyRegistry:
    def direct_lunit_client() -> LunitClient:
        return LunitClient(
            base_url=settings.lunit_api_url,
            api_key=settings.lunit_api_key,
            model=settings.lunit_model,
            timeout=settings.l2_timeout_seconds,
        )

    def lunit_client() -> LunitClient | L2PlusClient:
        if settings.l2_backend == "direct_l2":
            return direct_lunit_client()
        if settings.l2_backend == "l2_plus":
            return L2PlusClient(
                base_url=settings.lunit_api_url,
                api_key=settings.lunit_api_key,
                model=settings.lunit_model,
                timeout=settings.l2_timeout_seconds,
            )
        raise ValueError(
            f"unknown L2_BACKEND '{settings.l2_backend}'; choose direct_l2 or l2_plus"
        )

    def sol_client() -> SolClient:
        return SolClient(
            base_url=settings.openai_api_url,
            api_key=settings.openai_api_key,
            model=settings.sol_model,
            reasoning_effort=settings.sol_reasoning_effort,
            timeout=settings.sol_timeout_seconds,
        )

    registry = StrategyRegistry()
    registry.register("direct_l2", lambda: DirectL2Strategy(lunit_client()))
    registry.register("baseline_l2", lambda: DirectL2Strategy(direct_lunit_client()))
    for name in ("demographics", "demographics_sol"):
        registry.register(
            name,
            lambda: DemographicsStrategy(
                lunit=lunit_client(), sol=sol_client(), profile_count=settings.profile_count,
                max_parallel=settings.max_parallel_l2_calls,
                minimum_successes=settings.min_successful_profiles,
                pipeline_timeout=settings.pipeline_timeout_seconds,
            ),
        )
    for name in ("siusiubeom", "siusiubeom_h4"):
        registry.register(
            name,
            lambda: SiusiubeomH4Strategy(local_first=settings.l2_backend == "l2_plus"),
        )
    for name in ("patient-sim", "patient_sim"):
        registry.register(
            name,
            lambda: PatientSimStrategy(
                patient_simulator=PatientSimulatorClient(
                    base_url=settings.patient_simulator_url,
                    api_key=settings.lunit_api_key,
                    model=settings.patient_simulator_model,
                    timeout=settings.l2_timeout_seconds,
                ),
                lunit=lunit_client(), sol=sol_client(),
                pipeline_timeout=settings.pipeline_timeout_seconds,
            ),
        )
    registry.register(
        "multi_patient",
        lambda: MultiPatientStrategy(
            lunit=lunit_client(), sol=sol_client(), profile_count=settings.profile_count,
            max_parallel=settings.max_parallel_l2_calls,
            minimum_successes=settings.min_successful_profiles,
            pipeline_timeout=settings.pipeline_timeout_seconds,
        ),
    )
    registry.register(
        "multi_patient_sol",
        lambda: MultiPatientStrategy(
            lunit=lunit_client(), sol=sol_client(), profile_count=settings.profile_count,
            max_parallel=settings.max_parallel_l2_calls,
            minimum_successes=settings.min_successful_profiles,
            pipeline_timeout=settings.pipeline_timeout_seconds,
        ),
    )
    return registry

import json
import os
import platform
import sys
from pathlib import Path


def write_message(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def fail(message, error_type="RuntimeError"):
    write_message({"ok": False, "error": message, "type": error_type})
    raise RuntimeError(message)


def find_dwsim_automation_dir():
    search_roots = [
        Path(os.environ["DWSIM_AUTOMATION_DIR"]).expanduser()
        for _ in [0]
        if os.environ.get("DWSIM_AUTOMATION_DIR")
    ] + [
        Path("/Applications/DWSIM.app/Contents/MonoBundle"),
        Path("/Applications/DWSIM.app"),
        Path.home() / "Applications" / "DWSIM.app" / "Contents" / "MonoBundle",
        Path.home() / "Applications" / "DWSIM.app",
    ]

    for root in search_roots:
        if root.is_file() and root.name == "DWSIM.Automation.dll":
            return root.parent
        if not root.is_dir():
            continue

        direct_match = root / "DWSIM.Automation.dll"
        if direct_match.is_file():
            return root

        recursive_matches = sorted(root.glob("**/DWSIM.Automation.dll"))
        if recursive_matches:
            return recursive_matches[0].parent

    searched = "\n".join(str(path) for path in search_roots)
    fail(
        "Could not locate DWSIM.Automation.dll. Checked:\n" + searched,
        error_type="FileNotFoundError",
    )


def find_mono_root(workspace_dir):
    candidates = [
        Path(os.environ["MONO_ROOT"]).expanduser()
        for _ in [0]
        if os.environ.get("MONO_ROOT")
    ] + [
        workspace_dir / ".mono-runtime-x64" / "mono" / "6.14.1",
        Path("/Library/Frameworks/Mono.framework/Versions/Current"),
        Path("/usr/local"),
    ]

    for root in candidates:
        libmono = root / "lib" / "libmonosgen-2.0.dylib"
        assembly_dir = root / "lib"
        config_dir = root / "etc"
        if libmono.is_file() and assembly_dir.is_dir() and config_dir.is_dir():
            return libmono, assembly_dir, config_dir

    searched = "\n".join(str(path) for path in candidates)
    fail(
        "Could not locate a usable x86_64 Mono runtime. Checked:\n" + searched,
        error_type="FileNotFoundError",
    )


def require_flowsheet_path(argv):
    if len(argv) > 1:
        candidate = Path(argv[1]).expanduser().resolve()
        if candidate.is_file():
            return candidate
        fail(f"Flowsheet file not found: {candidate}", error_type="FileNotFoundError")

    fail("A flowsheet path argument is required.", error_type="FileNotFoundError")


def require_object(simulation, name):
    for getter in ("GetObject", "GetFlowsheetSimulationObject"):
        method = getattr(simulation, getter, None)
        if method is None:
            continue
        obj = method(name)
        if obj is None:
            continue
        return obj.GetAsObject() if hasattr(obj, "GetAsObject") else obj

    graphic_objects = getattr(simulation, "GraphicObjects", None)
    if graphic_objects is not None:
        for graphic_object in graphic_objects.Values:
            tag = getattr(graphic_object, "Tag", None)
            if isinstance(tag, str) and tag.lower() == name.lower():
                method = getattr(simulation, "GetObject", None)
                if method is not None:
                    obj = method(tag)
                    if obj is not None:
                        return obj.GetAsObject() if hasattr(obj, "GetAsObject") else obj
    raise ValueError(f"Flowsheet object not found: {name}")


def set_stream_composition(stream, composition):
    import System

    stream.OverallComposition = System.Array[float]([float(value) for value in composition])


def get_selected_compound_names(simulation, stream=None):
    candidates = []
    if stream is not None:
        candidates.extend(
            getattr(stream, attr, None)
            for attr in ("SelectedCompounds", "Compounds")
        )
    candidates.extend(
        getattr(simulation, attr, None)
        for attr in ("SelectedCompounds", "Compounds")
    )

    for compounds in candidates:
        if compounds is None:
            continue
        if hasattr(compounds, "Keys"):
            return [str(key).strip().lower() for key in compounds.Keys]
        try:
            return [
                str(getattr(compound, "Name", None) or getattr(compound, "ID", None) or compound).strip().lower()
                for compound in compounds
            ]
        except TypeError:
            continue

    return []


def build_feed_composition(simulation, stream, feed_ratio):
    reacting_fraction = 0.98
    n2_fraction = reacting_fraction / (1.0 + feed_ratio)
    h2_fraction = reacting_fraction - n2_fraction
    fractions = {
        "hydrogen": h2_fraction,
        "ammonia": 0.0,
        "nitrogen": n2_fraction,
        "argon": 0.01,
        "methane": 0.01,
    }

    compound_names = get_selected_compound_names(simulation, stream)

    if compound_names:
        return [fractions.get(name, 0.0) for name in compound_names]

    return [h2_fraction, 0.0, n2_fraction, 0.01, 0.01]


def calculate_flowsheet(interface, simulation):
    candidates = (
        lambda: interface.CalculateFlowsheet2(simulation),
        lambda: interface.CalculateFlowsheet(simulation),
        lambda: simulation.CalculateFlowsheet2(),
    )

    last_error = None
    for candidate in candidates:
        try:
            return candidate()
        except TypeError as error:
            last_error = error

    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to execute DWSIM flowsheet calculation.")


def get_component_fraction(simulation, stream, component_name):
    if hasattr(stream, "GetOverallComposition"):
        try:
            return float(stream.GetOverallComposition(component_name))
        except TypeError:
            composition = stream.GetOverallComposition()
        else:
            return float(stream.GetOverallComposition(component_name))
    else:
        composition = getattr(stream, "OverallComposition", None)

    names = get_selected_compound_names(simulation, stream)
    if composition is not None and names:
        index = names.index(component_name.strip().lower())
        return float(composition[index])

    raise AttributeError(f"Unable to read component fraction for {component_name}.")


def set_property(obj, candidates, value):
    for name in candidates:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return
    raise AttributeError(f"Expected one of {candidates} on object {obj}.")


def set_splitter_ratios(splitter, purge_fraction):
    for name in ("SplitRatios", "Ratios"):
        if hasattr(splitter, name):
            ratios = getattr(splitter, name)
            ratios[0] = purge_fraction
            ratios[1] = 1.0 - purge_fraction
            return
    raise AttributeError("Splitter object does not expose a writable ratios collection.")


def get_mass_flow(stream):
    if hasattr(stream, "GetMassFlow"):
        return float(stream.GetMassFlow())
    if hasattr(stream, "MassFlow"):
        return float(stream.MassFlow)
    raise AttributeError("Unable to read mass flow from stream.")


def collect_observation(simulation, reactor_temp_c, loop_pressure_bar):
    product_stream = require_object(simulation, "STR_Product_NH3")
    recycle_stream = require_object(simulation, "STR_Vapor_Recycle")
    energy_streams = [
        require_object(simulation, "EN_Comp1"),
        require_object(simulation, "EN_Comp2"),
        require_object(simulation, "EN_Comp3"),
        require_object(simulation, "EN_Recycle_Comp"),
    ]

    nh3_flow = get_mass_flow(product_stream)
    total_kw = float(sum(float(stream.EnergyFlow) for stream in energy_streams))
    methane = get_component_fraction(simulation, recycle_stream, "Methane")
    argon = get_component_fraction(simulation, recycle_stream, "Argon")
    return [nh3_flow, total_kw, float(reactor_temp_c), float(loop_pressure_bar), methane + argon]


def calculate_reward(simulation):
    product_stream = require_object(simulation, "STR_Product_NH3")
    energy_streams = [
        require_object(simulation, "EN_Comp1"),
        require_object(simulation, "EN_Comp2"),
        require_object(simulation, "EN_Comp3"),
        require_object(simulation, "EN_Recycle_Comp"),
    ]
    condenser_energy_stream = require_object(simulation, "EN_Condenser")
    preheater_energy_stream = require_object(simulation, "EN_Preheater")

    nh3_kg_per_s = get_mass_flow(product_stream)
    nh3_ton_per_h = nh3_kg_per_s * 3.6
    revenue = nh3_ton_per_h * 500.0
    total_kw = float(sum(float(stream.EnergyFlow) for stream in energy_streams))
    condenser_kw = float(condenser_energy_stream.EnergyFlow)
    preheater_kw = float(preheater_energy_stream.EnergyFlow)

    cost_comp_electricity = total_kw * 0.06
    cost_refrigeration = condenser_kw * 0.15
    cost_heating = preheater_kw * 0.04
    total_cost = cost_comp_electricity + cost_refrigeration + cost_heating
    reward = revenue - total_cost
    return reward, {
        "nh3_kg_per_s": nh3_kg_per_s,
        "nh3_ton_per_h": nh3_ton_per_h,
        "power_kw": total_kw,
        "condenser_kw": condenser_kw,
        "preheater_kw": preheater_kw,
        "revenue_per_h": revenue,
        "compressor_cost_per_h": cost_comp_electricity,
        "refrigeration_cost_per_h": cost_refrigeration,
        "heating_cost_per_h": cost_heating,
        "energy_cost_per_h": total_cost,
    }


def main():
    workspace_dir = Path.cwd().resolve()

    if sys.version_info[:2] != (3, 11):
        fail("The DWSIM helper requires Python 3.11.x.")

    if platform.machine() != "x86_64":
        fail("The DWSIM helper must run under Rosetta x86_64 on this Mac.")

    flowsheet_path = require_flowsheet_path(sys.argv)
    automation_dir = find_dwsim_automation_dir()
    libmono, assembly_dir, config_dir = find_mono_root(workspace_dir)

    try:
        from pythonnet import load
    except ImportError as exc:
        fail(str(exc), error_type=exc.__class__.__name__)

    load(
        "mono",
        libmono=str(libmono),
        assembly_dir=str(assembly_dir),
        config_dir=str(config_dir),
    )

    import clr

    if str(automation_dir) not in sys.path:
        sys.path.append(str(automation_dir))

    clr.AddReference(str(automation_dir / "DWSIM.Automation.dll"))
    clr.AddReference(str(automation_dir / "CapeOpen.dll"))
    clr.AddReference(str(automation_dir / "DWSIM.Interfaces.dll"))
    clr.AddReference(str(automation_dir / "DWSIM.GlobalSettings.dll"))

    from DWSIM.Automation import Automation3

    interface = Automation3()
    simulation = interface.LoadFlowsheet(str(flowsheet_path))

    write_message(
        {
            "ok": True,
            "status": "ready",
            "architecture": platform.machine(),
            "flowsheet": str(flowsheet_path),
        }
    )

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            command = request.get("command")

            if command == "ping":
                write_message({"ok": True, "status": "ready"})
                continue

            if command == "shutdown":
                write_message({"ok": True, "status": "shutting-down"})
                return

            if command == "reset":
                simulation = interface.LoadFlowsheet(str(flowsheet_path))
                calculate_flowsheet(interface, simulation)
                write_message({"ok": True, "status": "reset"})
                continue

            if command != "evaluate":
                raise ValueError(f"Unsupported command: {command}")

            action = [float(value) for value in request["action"]]
            (
                feed_ratio,
                stage1_bar,
                stage2_bar,
                loop_bar,
                reactor_temp_c,
                condenser_temp_c,
                purge_fraction,
            ) = action

            fresh_feed = require_object(simulation, "STR_FreshFeed")
            comp_stage1 = require_object(simulation, "COMP_Stage1")
            comp_stage2 = require_object(simulation, "COMP_Stage2")
            comp_stage3 = require_object(simulation, "COMP_Stage3")
            comp_recycle = require_object(simulation, "COMP_Recycle")
            heat_preheater = require_object(simulation, "HEAT_Preheater")
            reactor = require_object(simulation, "PFR_Reactor")
            condenser = require_object(simulation, "COOL_Condenser")
            purge_splitter = require_object(simulation, "SPLIT_Purge")

            set_stream_composition(fresh_feed, build_feed_composition(simulation, fresh_feed, feed_ratio))

            set_property(comp_stage1, ["POut"], stage1_bar * 100000.0)
            set_property(comp_stage2, ["POut"], stage2_bar * 100000.0)
            loop_pressure_pa = loop_bar * 100000.0
            set_property(comp_stage3, ["POut"], loop_pressure_pa)
            set_property(comp_recycle, ["POut"], loop_pressure_pa)

            reactor_temp_k = reactor_temp_c + 273.15
            set_property(heat_preheater, ["TOut", "OutletTemperature"], reactor_temp_k)
            set_property(reactor, ["Temperature", "OutletTemperature"], reactor_temp_k)
            set_property(condenser, ["TOut", "OutletTemperature"], condenser_temp_c + 273.15)

            set_splitter_ratios(purge_splitter, purge_fraction)

            errors = calculate_flowsheet(interface, simulation)
            if errors is not None and len(errors) > 0:
                raise RuntimeError("; ".join(str(error) for error in errors))

            reward, info = calculate_reward(simulation)
            observation = collect_observation(simulation, reactor_temp_c, loop_bar)
            write_message(
                {
                    "ok": True,
                    "reward": float(reward),
                    "observation": observation,
                    "terminated": False,
                    "truncated": False,
                    "info": info,
                }
            )
        except Exception as exc:
            write_message({"ok": False, "error": str(exc), "type": exc.__class__.__name__})


if __name__ == "__main__":
    main()
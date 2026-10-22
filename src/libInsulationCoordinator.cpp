#include <iostream>
#include <vector>
#include "json.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include "constructive_models/Insulation.h"
#include "constructive_models/MasMigration.h"
#include <MAS.hpp>
#include "processors/Inputs.h"


using namespace MAS;
using namespace emscripten;
using json = nlohmann::json;

std::string calculate_insulation(std::string inputsString){
    // On failure the result carries ONLY errorMessage: no distance field is
    // present unless it was computed. A pre-filled 0.0 here reads as "no
    // separation required" on a safety tool (ABT #1228).
    json result;
    try
    {
        // Build Inputs via the default ctor + from_json so we bypass
        // check_integrity(); the four insulation calculators read voltage
        // processed peak/rms + altitude + insulation requirements, and do
        // not need a synthesised magnetizing current (which check_integrity
        // tries to build and throws when prerequisites are missing).
        auto j = json::parse(inputsString);
        OpenMagnetics::compat::migrate_pre_1_0(j);
        OpenMagnetics::Inputs inputs;
        from_json(j, inputs);

        auto insulationCoordinator = OpenMagnetics::InsulationCoordinator();
        // One call, one success-or-throw for all four distances.
        json computed = insulationCoordinator.calculate_insulation_coordination(inputs);
        // MAS takes an absent key, never null, for an unset optional.
        for (auto it = computed.begin(); it != computed.end();) {
            if (it->is_null()) {
                it = computed.erase(it);
            }
            else {
                ++it;
            }
        }
        result = computed;
    }
    catch(const std::exception& ex)
    {
        result = json::object();
        result["errorMessage"] = ex.what();
    }
    return result.dump(4);
}


EMSCRIPTEN_BINDINGS(my_bindings) {
    function("calculate_insulation", &calculate_insulation);
}
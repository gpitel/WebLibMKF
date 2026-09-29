#include <iostream>
#include <vector>
#include "json.hpp"

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include "Constants.h"
#include "Definitions.h"
#include "constructive_models/Insulation.h"
#include "Defaults.h"
#include <MAS.hpp>
#include "constructive_models/Coil.h"
#include "constructive_models/NumberTurns.h"
#include "constructive_models/CorePiece.h"
#include "processors/MagneticSimulator.h"
#include "physical_models/WindingOhmicLosses.h"
#include "physical_models/WindingSkinEffectLosses.h"
#include "physical_models/WindingLosses.h"
#include "physical_models/Temperature.h"
#include "advisers/WireAdviser.h"
#include "advisers/CoilAdviser.h"
#include "advisers/CoreAdviser.h"
#include "advisers/MagneticAdviser.h"
#include "advisers/MagneticCombinator.h"  // virtual magnetics: re-wire a real transformer's windings
#include "support/LibraryContext.h"
#include "processors/Inputs.h"
#include "constructive_models/Core.h"
#include "physical_models/ComplexPermeability.h"
#include "physical_models/CoreLosses.h"
#include "physical_models/CoreTemperature.h"
#include "physical_models/InitialPermeability.h"
#include "physical_models/LeakageInductance.h"
#include "physical_models/MagnetizingInductance.h"
#include "physical_models/Inductance.h"
#include "physical_models/StrayCapacitance.h"
#include "physical_models/Reluctance.h"
#include "support/Painter.h"
#include "support/Utils.h"
#include "processors/Sweeper.h"
#include "processors/CircuitSimulatorInterface.h"
#include <magic_enum.hpp>


using namespace MAS;
using namespace emscripten;
using json = nlohmann::json;

// Forward declarations for waveform repetition helpers
void repeat_waveform_for_periods(std::vector<double>& time, std::vector<double>& data, size_t numberOfPeriods);
void repeat_operating_points_waveforms(json& operatingPoints, size_t numberOfPeriods);
void repeat_converter_waveforms_periods(json& converterWaveforms, size_t numberOfPeriods);
using ordered_json = nlohmann::ordered_json;

// Forward declarations for new converter processing functions

// Default frequency constant (100 kHz)
constexpr double DEFAULT_FREQUENCY_HZ = 100000.0;

// Helper function to process current signal descriptor if it lacks processed data
// Returns true if processing succeeded or was not needed, false on error
bool ensure_current_processed(SignalDescriptor& current, const std::string& functionName) {
    if (current.get_processed() && current.get_processed()->get_rms()) {
        return true;  // Already has processed data
    }
    
    if (!current.get_waveform()) {
        std::cerr << "Error in " << functionName << ": Current has no waveform to process" << std::endl;
        return false;
    }
    
    // Extract frequency from waveform time data or use default
    double frequency = DEFAULT_FREQUENCY_HZ;
    auto waveform = current.get_waveform().value();
    if (waveform.get_time() && waveform.get_time()->size() > 1) {
        auto time = waveform.get_time().value();
        double period = time.back() - time.front();
        if (period > 0) {
            frequency = 1.0 / period;
        }
    }
    
    // Process the current: first calculate harmonics, then processed data
    auto sampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(waveform, frequency);
    auto harmonics = OpenMagnetics::Inputs::calculate_harmonics_data(sampledWaveform, frequency);
    current.set_harmonics(harmonics);
    auto processed = OpenMagnetics::Inputs::calculate_processed_data(harmonics, sampledWaveform, true);
    current.set_processed(processed);
    
    return true;
}

// Helper function to process current signal descriptor if it lacks effective_frequency
// Returns true if processing succeeded or was not needed, false on error
bool ensure_current_processed_for_effective_frequency(SignalDescriptor& current, const std::string& functionName) {
    if (current.get_processed() && current.get_processed()->get_effective_frequency()) {
        return true;  // Already has processed data with effective frequency
    }
    
    if (!current.get_waveform()) {
        std::cerr << "Error in " << functionName << ": Current has no waveform to process" << std::endl;
        return false;
    }
    
    // Extract frequency from waveform time data or use default
    double frequency = DEFAULT_FREQUENCY_HZ;
    auto waveform = current.get_waveform().value();
    if (waveform.get_time() && waveform.get_time()->size() > 1) {
        auto time = waveform.get_time().value();
        double period = time.back() - time.front();
        if (period > 0) {
            frequency = 1.0 / period;
        }
    }
    
    // Process the current: first calculate harmonics, then processed data
    auto sampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(waveform, frequency);
    auto harmonics = OpenMagnetics::Inputs::calculate_harmonics_data(sampledWaveform, frequency);
    current.set_harmonics(harmonics);
    auto processed = OpenMagnetics::Inputs::calculate_processed_data(harmonics, sampledWaveform, true);
    current.set_processed(processed);
    
    return true;
}

std::map<std::string, double> get_constants() {
    std::map<std::string, double> constantsMap;
    constantsMap["residualGap"] = OpenMagnetics::constants.residualGap;
    constantsMap["minimumNonResidualGap"] = OpenMagnetics::constants.minimumNonResidualGap;
    constantsMap["vacuumPermeability"] = OpenMagnetics::constants.vacuumPermeability;
    constantsMap["vacuumPermittivity"] = OpenMagnetics::constants.vacuumPermittivity;
    constantsMap["quasiStaticFrequencyLimit"] = OpenMagnetics::constants.quasiStaticFrequencyLimit;
    constantsMap["spacerProtudingPercentage"] = OpenMagnetics::constants.spacerProtudingPercentage;
    constantsMap["coilPainterScale"] = OpenMagnetics::constants.coilPainterScale;
    constantsMap["numberPointsSampledWaveforms"] = OpenMagnetics::constants.numberPointsSampledWaveforms;
    constantsMap["minimumDistributedFringingFactor"] = OpenMagnetics::constants.minimumDistributedFringingFactor;
    constantsMap["maximumDistributedFringingFactor"] = OpenMagnetics::constants.maximumDistributedFringingFactor;
    constantsMap["initialGapLengthForSearching"] = OpenMagnetics::constants.initialGapLengthForSearching;
    constantsMap["roshenMagneticFieldStrengthStep"] = OpenMagnetics::constants.roshenMagneticFieldStrengthStep;
    constantsMap["foilToSectionMargin"] = OpenMagnetics::constants.foilToSectionMargin;
    constantsMap["planarToSectionMargin"] = OpenMagnetics::constants.planarToSectionMargin;
    return constantsMap;
}

std::map<std::string, double> get_defaults() {
    std::map<std::string, double> defaultsMap;
    defaultsMap["maximumProportionMagneticFluxDensitySaturation"] = OpenMagnetics::defaults.maximumProportionMagneticFluxDensitySaturation;
    defaultsMap["coreAdviserFrequencyReference"] = OpenMagnetics::defaults.coreAdviserFrequencyReference;
    defaultsMap["coreAdviserMagneticFluxDensityReference"] = OpenMagnetics::defaults.coreAdviserMagneticFluxDensityReference;
    defaultsMap["coreAdviserThresholdValidity"] = OpenMagnetics::defaults.coreAdviserThresholdValidity;
    defaultsMap["coreAdviserMaximumCoreTemperature"] = OpenMagnetics::defaults.coreAdviserMaximumCoreTemperature;
    defaultsMap["coreAdviserMaximumPercentagePowerCoreLosses"] = OpenMagnetics::defaults.coreAdviserMaximumPercentagePowerCoreLosses;
    defaultsMap["coreAdviserMaximumMagneticsAfterFiltering"] = OpenMagnetics::defaults.coreAdviserMaximumMagneticsAfterFiltering;
    defaultsMap["coreAdviserMaximumNumberStacks"] = OpenMagnetics::defaults.coreAdviserMaximumNumberStacks;
    defaultsMap["maximumCurrentDensity"] = OpenMagnetics::defaults.maximumCurrentDensity;
    defaultsMap["maximumEffectiveCurrentDensity"] = OpenMagnetics::defaults.maximumEffectiveCurrentDensity;
    defaultsMap["maximumNumberParallels"] = OpenMagnetics::defaults.maximumNumberParallels;
    defaultsMap["magneticFluxDensitySaturation"] = OpenMagnetics::defaults.magneticFluxDensitySaturation;
    defaultsMap["magnetizingInductanceThresholdValidity"] = OpenMagnetics::defaults.magnetizingInductanceThresholdValidity;
    defaultsMap["harmonicAmplitudeThreshold"] = OpenMagnetics::defaults.harmonicAmplitudeThreshold;
    defaultsMap["ambientTemperature"] = OpenMagnetics::defaults.ambientTemperature;
    defaultsMap["measurementFrequency"] = OpenMagnetics::defaults.measurementFrequency;
    defaultsMap["magneticFieldMirroringDimension"] = OpenMagnetics::defaults.magneticFieldMirroringDimension;
    defaultsMap["maximumCoilPattern"] = OpenMagnetics::defaults.maximumCoilPattern;
    defaultsMap["overlappingFactorSurroundingTurns"] = OpenMagnetics::defaults.overlappingFactorSurroundingTurns;

    return defaultsMap;
}

std::string standardize_signal_descriptor(std::string signalDescriptorString, double frequency) {
    try {
        SignalDescriptor signalDescriptor(json::parse(signalDescriptorString));

        auto standardSignalDescriptor = OpenMagnetics::Inputs::standardize_waveform(signalDescriptor, frequency);
        if (standardSignalDescriptor.get_harmonics()) {
            auto processed = OpenMagnetics::Inputs::calculate_processed_data(standardSignalDescriptor.get_harmonics().value(), standardSignalDescriptor.get_waveform().value(), true);
            standardSignalDescriptor.set_processed(processed);
        }
        else {
            auto processed = OpenMagnetics::Inputs::calculate_processed_data(standardSignalDescriptor.get_waveform().value(), frequency, true);
            standardSignalDescriptor.set_processed(processed);
        }

        json result;
        to_json(result, standardSignalDescriptor);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::vector<size_t> get_main_harmonic_indexes(std::string harmonicsString, double windingLossesHarmonicAmplitudeThreshold, int mainHarmonicIndex) {
    try {
        Harmonics harmonics;
        from_json(json::parse(harmonicsString), harmonics);

        std::vector<size_t> mainHarmonicIndexes;
        if (mainHarmonicIndex == -1) {
            mainHarmonicIndexes = OpenMagnetics::get_main_harmonic_indexes(harmonics, windingLossesHarmonicAmplitudeThreshold);
        }
        else {
            mainHarmonicIndexes = OpenMagnetics::get_main_harmonic_indexes(harmonics, windingLossesHarmonicAmplitudeThreshold, mainHarmonicIndex);
        }

        return mainHarmonicIndexes;
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return {0};
    }
}

std::vector<size_t> get_excitation_harmonic_indexes(std::string excitationString, double windingLossesHarmonicAmplitudeThreshold) {
    try {
        OperatingPointExcitation excitation(json::parse(excitationString));

        auto mainHarmonicIndexes = OpenMagnetics::get_excitation_harmonic_indexes(excitation, windingLossesHarmonicAmplitudeThreshold);

        return mainHarmonicIndexes;
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return {0};
    }
}

std::string calculate_harmonics(std::string waveformString, double frequency) {
    try {
        Waveform waveform;
        from_json(json::parse(waveformString), waveform);

        auto sampledCurrentWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(waveform, frequency);
        auto harmonics = OpenMagnetics::Inputs::calculate_harmonics_data(sampledCurrentWaveform, frequency);

        json result;
        to_json(result, harmonics);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_processed(std::string harmonicsString, std::string waveformString) {
    try {
        Waveform waveform;
        Harmonics harmonics;
        from_json(json::parse(waveformString), waveform);
        from_json(json::parse(harmonicsString), harmonics);

        auto processed = OpenMagnetics::Inputs::calculate_processed_data(harmonics, waveform, true);

        json result;
        to_json(result, processed);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << harmonicsString << std::endl;
        std::cerr << waveformString << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_core_data_from_shape(std::string shapeString){
    try {
        CoreShape shape(json::parse(shapeString));
        OpenMagnetics::Core core;
        CoreFunctionalDescription coreFunctionalDescription;
        coreFunctionalDescription.set_shape(shape);
        coreFunctionalDescription.set_material("Dummy");
        coreFunctionalDescription.set_number_stacks(1);
        if (shape.get_magnetic_circuit() == MagneticCircuit::OPEN) {
            coreFunctionalDescription.set_type(CoreType::TWO_PIECE_SET);
        }
        else {
            if (shape.get_family() == CoreShapeFamily::T) {
                coreFunctionalDescription.set_type(CoreType::TOROIDAL);
            }
            else {
                coreFunctionalDescription.set_type(CoreType::CLOSED_SHAPE);
            }
        }
        core.set_functional_description(coreFunctionalDescription);
        core.process_data();

        json result;
        to_json(result, core);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_all_core_data_from_shapes(){
    try {
        // Get all shapes from the database
        auto allShapes = OpenMagnetics::get_shapes(true);
        
        json resultArray = json::array();
        
        for (const auto& shape : allShapes) {
            try {
                OpenMagnetics::Core core;
                CoreFunctionalDescription coreFunctionalDescription;
                coreFunctionalDescription.set_shape(shape);
                coreFunctionalDescription.set_material("Dummy");
                coreFunctionalDescription.set_number_stacks(1);
                if (shape.get_magnetic_circuit() == MagneticCircuit::OPEN) {
                    coreFunctionalDescription.set_type(CoreType::TWO_PIECE_SET);
                }
                else {
                    if (shape.get_family() == CoreShapeFamily::T) {
                        coreFunctionalDescription.set_type(CoreType::TOROIDAL);
                    }
                    else {
                        coreFunctionalDescription.set_type(CoreType::CLOSED_SHAPE);
                    }
                }
                core.set_functional_description(coreFunctionalDescription);
                core.process_data();
                
                json coreJson;
                to_json(coreJson, core);
                resultArray.push_back(coreJson);
            }
            catch (const std::exception &exc) {
                // Skip shapes that fail to process, log if needed
                // std::cerr << "Failed to process shape: " << exc.what() << std::endl;
            }
        }
        
        return resultArray.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_core_data(std::string coreDataString, bool includeMaterialData){
    try {
        OpenMagnetics::Core core(json::parse(coreDataString), includeMaterialData, true);

        json result;
        to_json(result, core);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_bobbin_data(std::string magneticString){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        auto optionalBobbin = magnetic.get_coil().get_bobbin();
        OpenMagnetics::Bobbin bobbin;

        if (std::holds_alternative<std::string>(optionalBobbin)) {
            auto bobbinString = std::get<std::string>(optionalBobbin);
            if (bobbinString == "Dummy") {
                return "Exception: " + std::string{"Use create_simple_bobbin_from_core instead"};
            }
            else {
                bobbin = OpenMagnetics::find_bobbin_by_name(bobbinString);
            }
        }
        else {
            bobbin = OpenMagnetics::Bobbin(std::get<std::string>(optionalBobbin));
            bobbin.process_data();
        }

        json result;
        to_json(result, bobbin);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string create_simple_bobbin_from_core(std::string coresString){
    try {
        OpenMagnetics::Core core(json::parse(coresString), false, true);
        auto bobbin = OpenMagnetics::Bobbin::create_quick_bobbin(core);

        json result;
        to_json(result, bobbin);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string create_simple_bobbin_from_core_with_custom_thickness(std::string coresString, double thickness){
    try {
        OpenMagnetics::Core core(json::parse(coresString), false, true);
        auto bobbin = OpenMagnetics::Bobbin::create_quick_bobbin(core, thickness);

        json result;
        to_json(result, bobbin);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string create_simple_bobbin_from_core_with_custom_thicknesses(std::string coresString, double wallThickness, double columnThickness){
    try {
        OpenMagnetics::Core core(json::parse(coresString), false, true);
        auto bobbin = OpenMagnetics::Bobbin::create_quick_bobbin(core, wallThickness, columnThickness);

        json result;
        to_json(result, bobbin);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


std::string get_wire_data(std::string windingDataString){
    try {
        OpenMagnetics::Winding winding(json::parse(windingDataString));
        auto wire = OpenMagnetics::Coil::resolve_wire(winding);
        json result;
        to_json(result, wire);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_wire_data_by_name(std::string name){
    try {
        auto wireData = OpenMagnetics::find_wire_by_name(name);
        json result;
        to_json(result, wireData);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_wire_data_by_standard_name(std::string standardName){
    auto wires = OpenMagnetics::get_wires();
    for (auto wire : wires) {
        if (!wire.get_standard_name()) {
            continue;
        }
        if (wire.get_standard_name().value() == standardName) {
            auto coating = wire.resolve_coating();
            if (!coating) {
                continue;
            }
            if (!coating->get_grade()) {
                continue;
            }
            // // Hardcoded
            if (coating->get_grade().value() == 1) {
                json result;
                to_json(result, wire);
                return result.dump(4);
            }
        }
    }
    return "{}";
}

std::string get_planar_wire_by_standard_name(std::string standardName){
    try {
        // Normalize input: add period if missing (e.g., "2 oz" -> "2 oz.")
        std::string normalizedName = standardName;
        if (!normalizedName.empty() && normalizedName.back() != '.') {
            normalizedName += '.';
        }
        
        auto wires = OpenMagnetics::get_wires(WireType::PLANAR);
        for (auto wire : wires) {
            if (!wire.get_standard_name()) {
                continue;
            }
            std::string wireStandardName = wire.get_standard_name().value();
            // Try matching both normalized and original
            if (wireStandardName == standardName || wireStandardName == normalizedName) {
                json result;
                to_json(result, wire);
                return result.dump(4);
            }
        }
        return "{}";
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Helper function to repeat waveform data for multiple periods
void repeat_waveform_for_periods(std::vector<double>& time, std::vector<double>& data, size_t numberOfPeriods) {
    if (numberOfPeriods <= 1 || time.empty() || data.empty()) {
        return;
    }
    
    double period = time.back() - time.front();
    size_t originalSize = time.size();
    
    // Reserve space for repeated data
    time.reserve(originalSize * numberOfPeriods);
    data.reserve(originalSize * numberOfPeriods);
    
    for (size_t p = 1; p < numberOfPeriods; ++p) {
        double offset = p * period;
        for (size_t i = 0; i < originalSize; ++i) {
            // Skip first point of subsequent periods to avoid duplicates
            if (i == 0) continue;
            time.push_back(time[i] + offset);
            data.push_back(data[i]);
        }
    }
}

// Helper function to repeat all waveforms in operating points
void repeat_operating_points_waveforms(json& operatingPoints, size_t numberOfPeriods) {
    if (numberOfPeriods <= 1 || !operatingPoints.is_array()) {
        return;
    }
    
    for (auto& op : operatingPoints) {
        if (!op.contains("excitationsPerWinding") || !op["excitationsPerWinding"].is_array()) {
            continue;
        }
        
        for (auto& excitation : op["excitationsPerWinding"]) {
            // Repeat voltage waveform
            if (excitation.contains("voltage") && excitation["voltage"].contains("waveform")) {
                auto& waveform = excitation["voltage"]["waveform"];
                if (waveform.contains("time") && waveform.contains("data")) {
                    auto time = waveform["time"].get<std::vector<double>>();
                    auto data = waveform["data"].get<std::vector<double>>();
                    repeat_waveform_for_periods(time, data, numberOfPeriods);
                    waveform["time"] = time;
                    waveform["data"] = data;
                }
            }
            
            // Repeat current waveform
            if (excitation.contains("current") && excitation["current"].contains("waveform")) {
                auto& waveform = excitation["current"]["waveform"];
                if (waveform.contains("time") && waveform.contains("data")) {
                    auto time = waveform["time"].get<std::vector<double>>();
                    auto data = waveform["data"].get<std::vector<double>>();
                    repeat_waveform_for_periods(time, data, numberOfPeriods);
                    waveform["time"] = time;
                    waveform["data"] = data;
                }
            }
        }
    }
}

// Helper function to repeat all waveforms in converter waveforms
void repeat_converter_waveforms_periods(json& converterWaveforms, size_t numberOfPeriods) {
    if (numberOfPeriods <= 1 || !converterWaveforms.is_array()) {
        return;
    }
    
    for (auto& cw : converterWaveforms) {
        // Repeat input voltage waveform
        if (cw.contains("inputVoltage") && cw["inputVoltage"].contains("time") && cw["inputVoltage"].contains("data")) {
            auto time = cw["inputVoltage"]["time"].get<std::vector<double>>();
            auto data = cw["inputVoltage"]["data"].get<std::vector<double>>();
            repeat_waveform_for_periods(time, data, numberOfPeriods);
            cw["inputVoltage"]["time"] = time;
            cw["inputVoltage"]["data"] = data;
        }
        
        // Repeat input current waveform
        if (cw.contains("inputCurrent") && cw["inputCurrent"].contains("time") && cw["inputCurrent"].contains("data")) {
            auto time = cw["inputCurrent"]["time"].get<std::vector<double>>();
            auto data = cw["inputCurrent"]["data"].get<std::vector<double>>();
            repeat_waveform_for_periods(time, data, numberOfPeriods);
            cw["inputCurrent"]["time"] = time;
            cw["inputCurrent"]["data"] = data;
        }
        
        // Repeat output voltages
        if (cw.contains("outputVoltages") && cw["outputVoltages"].is_array()) {
            for (auto& outV : cw["outputVoltages"]) {
                if (outV.contains("time") && outV.contains("data")) {
                    auto time = outV["time"].get<std::vector<double>>();
                    auto data = outV["data"].get<std::vector<double>>();
                    repeat_waveform_for_periods(time, data, numberOfPeriods);
                    outV["time"] = time;
                    outV["data"] = data;
                }
            }
        }
        
        // Repeat output currents
        if (cw.contains("outputCurrents") && cw["outputCurrents"].is_array()) {
            for (auto& outI : cw["outputCurrents"]) {
                if (outI.contains("time") && outI.contains("data")) {
                    auto time = outI["time"].get<std::vector<double>>();
                    auto data = outI["data"].get<std::vector<double>>();
                    repeat_waveform_for_periods(time, data, numberOfPeriods);
                    outI["time"] = time;
                    outI["data"] = data;
                }
            }
        }
    }
}

double get_wire_outer_width_rectangular(double conductingWidth, int grade, std::string wireStandardString){
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_width_rectangular(conductingWidth, grade, wireStandard);
}

double get_wire_outer_height_rectangular(double conductingHeight, int grade, std::string wireStandardString){
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_height_rectangular(conductingHeight, grade, wireStandard);
}

double get_wire_outer_diameter_bare_litz(double conductingDiameter, int numberConductors, int grade, std::string wireStandardString) {
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_diameter_bare_litz(conductingDiameter, numberConductors, grade, wireStandard);
}

double get_wire_outer_diameter_served_litz(double conductingDiameter, int numberConductors, int grade, int numberLayers, std::string wireStandardString) {
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_diameter_served_litz(conductingDiameter, numberConductors, grade, numberLayers, wireStandard);
}

double get_wire_outer_diameter_insulated_litz(double conductingDiameter, int numberConductors, int numberLayers, double thicknessLayers, int grade, std::string wireStandardString) {
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_diameter_insulated_litz(conductingDiameter, numberConductors, numberLayers, thicknessLayers, grade, wireStandard);
}

double get_wire_outer_diameter_enamelled_round(double conductingDiameter, int grade, std::string wireStandardString) {
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_diameter_round(conductingDiameter, grade, wireStandard);
}

double get_wire_outer_diameter_insulated_round(double conductingDiameter, int numberLayers, double thicknessLayers, std::string wireStandardString) {
    WireStandard wireStandard;
    from_json(wireStandardString, wireStandard);
    return OpenMagnetics::Wire::get_outer_diameter_round(conductingDiameter, numberLayers, thicknessLayers, wireStandard);
}

// Helper: resolve a Wire from either a JSON string name (e.g. "\"Round S18A01FX-3\"")
// or a full JSON wire object.
OpenMagnetics::Wire resolve_wire_from_string(const std::string& wireString) {
    auto j = json::parse(wireString);
    if (j.is_string()) {
        // Wire is referenced by name — look it up in the database
        return OpenMagnetics::find_wire_by_name(j.get<std::string>());
    }
    else {
        // Wire is a full object
        return OpenMagnetics::Wire(j);
    }
}

std::vector<double> get_outer_dimensions(std::string wireString) {
    auto wire = resolve_wire_from_string(wireString);
    return {wire.get_maximum_outer_width(), wire.get_maximum_outer_height()};
}


std::string get_strand_by_standard_name(std::string standardName){
    auto wires = OpenMagnetics::get_wires();
    for (auto wire : wires) {
        if (!wire.get_standard_name()) {
            continue;
        }
        auto coating = wire.resolve_coating();
        if (!coating) {
            continue;
        }
        // We are looking for enamelled wires for strands
        if (coating->get_type() != InsulationWireCoatingType::ENAMELLED) {
            continue;
        }

        if (!coating->get_grade()) {
            throw std::runtime_error("Missing grade");
        }

        if (wire.get_standard_name().value() == standardName && coating->get_grade().value() == 1) {
            json result;
            to_json(result, wire);
            return result.dump(4);
        }
    }

    json result;
    result["errorMessage"] = "Wire not found by standard name";
    return result.dump(4);
}

double get_wire_conducting_diameter_by_standard_name(std::string standardName){
    auto wires = OpenMagnetics::get_wires();
    for (auto wire : wires) {
        if (!wire.get_standard_name()) {
            continue;
        }
        if (wire.get_standard_name().value() == standardName) {
            return OpenMagnetics::resolve_dimensional_values(wire.get_conducting_diameter().value());
        }
    }

    return -1;
}


std::string get_equivalent_wire(std::string oldWireString, std::string newWireTypeString, double effectivefrequency){
    try {
        OpenMagnetics::Wire oldWire(json::parse(oldWireString));
        WireType newWireType;
        from_json(json::parse(newWireTypeString), newWireType);

        auto newWire = OpenMagnetics::Wire::get_equivalent_wire(oldWire, newWireType, effectivefrequency);

        json result;
        to_json(result, newWire);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;

        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_coating_label(std::string wireString){
    try {
        OpenMagnetics::Wire wire(json::parse(wireString));
        auto coatingLabel = wire.encode_coating_label();
        return coatingLabel;
    }
    catch(const std::runtime_error& re)
    {
        return "Exception: " + std::string{re.what()};
    }
    catch(const std::exception& ex)
    {
        return "Exception: " + std::string{ex.what()};
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string get_wire_coating_by_label(std::string label){
    try {
        auto wires = OpenMagnetics::get_wires();
        InsulationWireCoating insulationWireCoating;
        for (auto wire : wires) {
            auto coatingLabel = wire.encode_coating_label();
            if (coatingLabel == label) {
                if (wire.resolve_coating()) {
                    insulationWireCoating = wire.resolve_coating().value();
                }
                else {
                    insulationWireCoating.set_type(InsulationWireCoatingType::BARE);
                }
                break;
            }
        }
        json result;
        to_json(result, insulationWireCoating);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::vector<std::string> get_coating_labels_by_type(std::string wireTypeString){
    WireType wireType(json::parse(wireTypeString));

    auto wires = OpenMagnetics::get_wires(wireType);

    std::vector<std::string> coatingLabels;
    for (auto wire : wires) {
        auto coatingLabel = wire.encode_coating_label();
        if (std::find(coatingLabels.begin(), coatingLabels.end(), coatingLabel) == coatingLabels.end()) {
            coatingLabels.push_back(coatingLabel);
        }
    }

    return coatingLabels;
}

std::string load_core_data(std::string coresString){
    try {
        json result = json::array();
        for (auto& coreJson : json::parse(coresString)) {
            OpenMagnetics::Core core(coreJson, false);
            json aux;
            to_json(aux, core);
            result.push_back(aux);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_material_data(std::string materialName){
    try {
        auto materialData = OpenMagnetics::find_core_material_by_name(materialName);
        json result;
        to_json(result, materialData);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_core_temperature_dependant_parameters(std::string coreData, double temperature){
    try {
        json coreJson = json::parse(coreData);
        MAS::CoreFunctionalDescription coreFunctionalDescription(coreJson["functionalDescription"]);
        MAS::CoreProcessedDescription coreProcessedDescription(coreJson["processedDescription"]);
        OpenMagnetics::Core core;
        core.set_functional_description(coreFunctionalDescription);
        core.set_processed_description(coreProcessedDescription);
        
        // Handle null or missing geometricalDescription
        if (coreJson.contains("geometricalDescription") && !coreJson["geometricalDescription"].is_null()) {
            std::vector<MAS::CoreGeometricalDescriptionElement> coreGeometricalDescription(coreJson["geometricalDescription"]);
            core.set_geometrical_description(coreGeometricalDescription);
        } else {
            core.set_geometrical_description(std::nullopt);
        }
        
        json result;

        result["magneticFluxDensitySaturation"] = core.get_magnetic_flux_density_saturation(temperature, false);
        result["magneticFieldStrengthSaturation"] = core.get_magnetic_field_strength_saturation(temperature);
        result["initialPermeability"] = core.get_initial_permeability(temperature);
        result["effectivePermeability"] = core.get_effective_permeability(temperature);
        // Core mass from the material density and the core volume (NaN when the record has no density).
        result["mass"] = core.get_mass();
        result["reluctance"] = core.get_reluctance(temperature);
        auto reluctanceModel = OpenMagnetics::ReluctanceModel::factory();
        result["permeance"] = 1.0 / reluctanceModel->get_ungapped_core_reluctance(core);
        result["resistivity"] = core.get_resistivity(temperature);

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_shape_data(std::string shapeName){
    try {
        auto shapeData = OpenMagnetics::find_core_shape_by_name(shapeName);

        json result;
        to_json(result, shapeData);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::vector<std::string> get_available_core_shape_families(){
    std::vector<std::string> families;
    for (auto& family : OpenMagnetics::get_core_shape_families()) {
        json familyJson;
        to_json(familyJson, family);
        families.push_back(familyJson);
    }
    return families;
}

// The families the ENGINE can build, which is not the same question as
// get_available_core_shape_families() above: that one reports the families present in the
// loaded shape database, so a buildable family that ships no bare-core record (MOLDED,
// DRUM_SEMISHIELDED — the construction is reconstructed per part, not sold as a core) never
// appeared in the UI at all. A catalogue browser must offer these; a custom shape in such a
// family is a perfectly ordinary part.
std::vector<std::string> get_supported_core_shape_families(){
    std::vector<std::string> families;
    for (auto& family : OpenMagnetics::get_supported_core_shape_families()) {
        json familyJson;
        to_json(familyJson, family);
        families.push_back(familyJson);
    }
    return families;
}

std::vector<std::string> get_available_core_manufacturers(){
    std::vector<std::string> manufacturers;
    auto materials = OpenMagnetics::get_materials("");
    for (auto material : materials) {
        std::string manufacturer = material.get_manufacturer_info().get_name();
        if (std::find(manufacturers.begin(), manufacturers.end(), manufacturer) == manufacturers.end()) {
            manufacturers.push_back(manufacturer);
        }
    }
    return manufacturers;
}

std::vector<std::string> get_available_core_materials(std::string manufacturer){
    return OpenMagnetics::get_core_material_names(manufacturer);
}

std::vector<std::string> get_available_core_materials_with_loss_model(std::string manufacturer){
    // Materials characterised only for interference suppression (complex
    // permeability, no volumetric/mass loss data) cannot feed any core-loss
    // computation. Power/filter design flows list materials through this
    // filtered variant so a selection can never end in MODEL_NOT_AVAILABLE;
    // common-mode-choke flows keep the unfiltered list above (ABT #401).
    std::vector<std::string> names;
    for (auto& material : OpenMagnetics::get_materials(manufacturer)) {
        if (!OpenMagnetics::CoreLossesModel::get_methods(material).empty()) {
            names.push_back(material.get_name());
        }
    }
    return names;
}

/**
 * Catalogue summary of every core material, for the web material picker
 * table (ABT #1072). One call instead of one get_material_data() per
 * material, and only the properties a designer filters on, each resolved by
 * MKF's own models (initial permeability, saturation, resistivity, losses)
 * rather than re-read from the raw MAS record in JavaScript.
 *
 * Numbers are in SI. Properties a material's record does not carry come back
 * as null, with the engine's reason in `missing` so the table can show it —
 * a blank cell is an explicit "no data", never a substituted value. The loss
 * reference point is the classic datasheet one: sinusoidal B̂ at the given
 * frequency and temperature, W/m³, using the material's first loss method.
 */
// One row per catalogue wire for the web's wire table (ABT #1110): the identity and the
// dimensions a designer filters on, every length in metres, absent data as null (never a
// substitute). Litz rows carry their strand; coating is resolved through Wire::resolve_coating.
std::string get_wires_summary() {
    try {
        auto dimensionOrNull = [](const std::optional<DimensionWithTolerance>& dimension) -> json {
            if (!dimension) {
                return nullptr;
            }
            return OpenMagnetics::resolve_dimensional_values(dimension.value());
        };
        json result = json::array();
        for (auto& wire : OpenMagnetics::get_wires()) {
            json row;
            if (!wire.get_name()) {
                throw std::runtime_error("A catalogue wire has no name");
            }
            row["name"] = wire.get_name().value();
            json typeJson;
            to_json(typeJson, wire.get_type());
            row["type"] = typeJson;
            if (wire.get_standard()) {
                json standardJson;
                to_json(standardJson, wire.get_standard().value());
                row["standard"] = standardJson;
            }
            else {
                row["standard"] = nullptr;
            }
            row["standardName"] = wire.get_standard_name() ? json(wire.get_standard_name().value()) : json(nullptr);
            row["manufacturer"] = (wire.get_manufacturer_info() && wire.get_manufacturer_info()->get_name() != "") ? json(wire.get_manufacturer_info()->get_name()) : json(nullptr);
            if (wire.get_material()) {
                if (std::holds_alternative<std::string>(wire.get_material().value())) {
                    row["material"] = std::get<std::string>(wire.get_material().value());
                }
                else {
                    row["material"] = std::get<WireMaterial>(wire.get_material().value()).get_name();
                }
            }
            else {
                row["material"] = nullptr;
            }
            row["numberConductors"] = wire.get_number_conductors() ? json(wire.get_number_conductors().value()) : json(nullptr);
            row["conductingDiameter"] = dimensionOrNull(wire.get_conducting_diameter());
            row["outerDiameter"] = dimensionOrNull(wire.get_outer_diameter());
            row["conductingWidth"] = dimensionOrNull(wire.get_conducting_width());
            row["conductingHeight"] = dimensionOrNull(wire.get_conducting_height());
            row["outerWidth"] = dimensionOrNull(wire.get_outer_width());
            row["outerHeight"] = dimensionOrNull(wire.get_outer_height());
            if (wire.get_type() == WireType::LITZ && wire.get_strand()) {
                auto strand = wire.resolve_strand();
                row["strandStandardName"] = strand.get_standard_name() ? json(strand.get_standard_name().value()) : json(nullptr);
                row["strandConductingDiameter"] = dimensionOrNull(strand.get_conducting_diameter());
            }
            else {
                row["strandStandardName"] = nullptr;
                row["strandConductingDiameter"] = nullptr;
            }
            auto coating = wire.resolve_coating();
            if (coating && coating->get_type()) {
                json coatingTypeJson;
                to_json(coatingTypeJson, coating->get_type().value());
                row["coatingType"] = coatingTypeJson;
                row["coatingGrade"] = coating->get_grade() ? json(coating->get_grade().value()) : json(nullptr);
                row["coatingLayers"] = coating->get_number_layers() ? json(coating->get_number_layers().value()) : json(nullptr);
            }
            else {
                row["coatingType"] = nullptr;
                row["coatingGrade"] = nullptr;
                row["coatingLayers"] = nullptr;
            }
            result.push_back(row);
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_core_materials_summary(double temperatureA, double temperatureB, double lossFrequency, double lossMagneticFluxDensityPeak, double lossTemperature){
    try {
        json result = json::array();
        for (auto& material : OpenMagnetics::get_materials("")) {
            json row;
            json missing = json::object();
            row["name"] = material.get_name();
            row["manufacturer"] = material.get_manufacturer_info().get_name();
            row["commercialName"] = material.get_commercial_name() ? json(material.get_commercial_name().value()) : json(nullptr);
            row["family"] = material.get_family() ? json(material.get_family().value()) : json(nullptr);
            {
                json materialType;
                to_json(materialType, material.get_material());
                row["materialType"] = materialType;
            }
            if (material.get_material_composition()) {
                json composition;
                to_json(composition, material.get_material_composition().value());
                row["composition"] = composition;
            }
            else {
                row["composition"] = nullptr;
            }
            {
                json applications = json::array();
                if (material.get_application()) {
                    for (auto& application : material.get_application().value()) {
                        json applicationJson;
                        to_json(applicationJson, application);
                        applications.push_back(applicationJson);
                    }
                }
                row["applications"] = applications;
            }
            row["curieTemperature"] = material.get_curie_temperature() ? json(material.get_curie_temperature().value()) : json(nullptr);
            row["density"] = material.get_density() ? json(material.get_density().value()) : json(nullptr);

            auto resolve = [&](const std::string& key, auto&& compute) {
                try {
                    row[key] = compute();
                }
                catch (const std::exception& exc) {
                    row[key] = nullptr;
                    missing[key] = std::string{exc.what()};
                }
            };
            resolve("initialPermeabilityA", [&]() { return OpenMagnetics::Core::get_initial_permeability(material, temperatureA); });
            resolve("initialPermeabilityB", [&]() { return OpenMagnetics::Core::get_initial_permeability(material, temperatureB); });
            resolve("saturationA", [&]() { return OpenMagnetics::Core::get_magnetic_flux_density_saturation(material, temperatureA, false); });
            resolve("saturationB", [&]() { return OpenMagnetics::Core::get_magnetic_flux_density_saturation(material, temperatureB, false); });
            resolve("resistivityA", [&]() { return OpenMagnetics::Core::get_resistivity(material, temperatureA); });

            auto methods = OpenMagnetics::CoreLossesModel::get_methods_string(material);
            row["lossMethods"] = methods;
            row["hasLossModel"] = !methods.empty();
            resolve("volumetricLossesReference", [&]() {
                auto lossMethods = OpenMagnetics::CoreLossesModel::get_methods(material);
                if (lossMethods.empty()) {
                    throw std::runtime_error("No core losses method available for material " + material.get_name());
                }
                auto model = OpenMagnetics::CoreLossesModel::factory(lossMethods[0]);
                auto waveform = OpenMagnetics::Inputs::create_waveform(WaveformLabel::SINUSOIDAL, 2 * lossMagneticFluxDensityPeak, lossFrequency);
                SignalDescriptor magneticFluxDensity;
                magneticFluxDensity.set_waveform(waveform);
                magneticFluxDensity.set_processed(OpenMagnetics::Inputs::calculate_processed_data(waveform, lossFrequency));
                OperatingPointExcitation excitation;
                excitation.set_frequency(lossFrequency);
                excitation.set_magnetic_flux_density(magneticFluxDensity);
                return model->get_core_volumetric_losses(material, excitation, lossTemperature);
            });

            row["missing"] = missing;
            result.push_back(row);
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::vector<std::string> get_available_core_shapes(){
    return OpenMagnetics::get_core_shape_names();
}

std::vector<std::string> get_available_core_shapes_by_manufacturer(std::string manufacturer){
    return OpenMagnetics::get_core_shape_names(manufacturer);
}

std::vector<std::string> get_available_core_shapes_by_family(std::string familyString){
    try {
        CoreShapeFamily family;
        from_json(familyString, family);
         
        return OpenMagnetics::get_core_shape_names(family);
    }
    catch (const std::exception &exc) {
        return {"Exception: " + std::string{exc.what()}};
    }
}

std::vector<std::string> get_shape_family_dimensions(std::string familyString, std::string familySubtype) {
    try {
        CoreShapeFamily family;
        from_json(familyString, family);

        std::vector<std::string> dimensions;
        if (familySubtype != "") {
            dimensions = OpenMagnetics::get_shape_family_dimensions(family, familySubtype);
        }
        else {
            dimensions = OpenMagnetics::get_shape_family_dimensions(family);
        }
         
        return dimensions;
    }
    catch (const std::exception &exc) {
        return {"Exception: " + std::string{exc.what()}};
    }
}

std::vector<std::string> get_shape_family_subtypes(std::string familyString) {
    try {
        CoreShapeFamily family;
        from_json(familyString, family);

        std::vector<std::string> familySubtypes;
        familySubtypes = OpenMagnetics::get_shape_family_subtypes(family);

        return familySubtypes;
    }
    catch (const std::exception &exc) {
        return {"Exception: " + std::string{exc.what()}};
    }
}

std::vector<std::string> get_available_wires(){
    return OpenMagnetics::get_wire_names();
}

std::vector<std::string> get_unique_wire_diameters(std::string wireStandardString){
    try {
        WireStandard wireStandard(json::parse(wireStandardString));
        std::vector<std::string> uniqueStandardName;

        {
            auto wires = OpenMagnetics::get_wires(WireType::ROUND, wireStandard);

            for (auto wire : wires) {
                if (!wire.get_standard_name()) {
                    continue;
                }
                auto standardName = wire.get_standard_name().value();
                if (std::find(uniqueStandardName.begin(), uniqueStandardName.end(), standardName) == uniqueStandardName.end()) {
                    uniqueStandardName.push_back(standardName);
                }
            }
        }

        {
            auto wires = OpenMagnetics::get_wires(WireType::LITZ, wireStandard);

            for (auto wire : wires) {
                auto strand = wire.resolve_strand();

                auto strandStandardName = strand.get_standard_name().value();
                if (std::find(uniqueStandardName.begin(), uniqueStandardName.end(), strandStandardName) == uniqueStandardName.end()) {
                    uniqueStandardName.push_back(strandStandardName);
                }
            }
        }
        return uniqueStandardName;
    }
    catch (const std::exception &exc) {
        return {"Exception: " + std::string{exc.what()}};
    }
}


std::vector<std::string> get_planar_thicknesses(){
    try {
        std::vector<std::string> uniqueStandardName;
        auto wires = OpenMagnetics::get_wires(WireType::PLANAR);

        for (auto wire : wires) {

            auto standardName = wire.get_standard_name().value();
            if (std::find(uniqueStandardName.begin(), uniqueStandardName.end(), standardName) == uniqueStandardName.end()) {
                uniqueStandardName.push_back(standardName);
            }
        }
        return uniqueStandardName;
    }
    catch (const std::exception &exc) {
        return {"Exception: " + std::string{exc.what()}};
    }
}

std::vector<std::string> get_available_wire_types(){
    // std::vector<std::string> wireTypes;

    // for (auto [value, name] : magic_enum::enum_entries<WireType>()) {
    //     json wireTypeString;
    //     if (value == WireType::PLANAR) {
    //         // TODO Add support for planar
    //         continue;
    //     }
    //     to_json(wireTypeString, value);
    //     wireTypes.push_back(wireTypeString);
    // }
    std::vector<MAS::WireType> wireTypes;
    for (auto [reference, wire] : OpenMagnetics::wireDatabase) {
        auto wireType = wire.get_type();
        if (wireType == WireType::PLANAR) {
            // TODO Add support for planar
            continue;
        }
        if (std::find(wireTypes.begin(), wireTypes.end(), wireType) == wireTypes.end()) {
            wireTypes.push_back(wireType);
        }
    }
    std::vector<std::string> wireTypesString;
    for (auto wireType : wireTypes) {
        json wireTypeString;        
        to_json(wireTypeString, wireType);
        wireTypesString.push_back(wireTypeString);
    }

    return wireTypesString;
}

std::vector<std::string> get_available_wire_standards(){
    // std::vector<std::string> wireStandards;

    // for (auto [value, name] : magic_enum::enum_entries<WireStandard>()) {
    //     json wireStandardString;
    //     to_json(wireStandardString, value);
    //     wireStandards.push_back(wireStandardString);
    // }

    std::vector<MAS::WireStandard> wireStandards;
    for (auto [reference, wire] : OpenMagnetics::wireDatabase) {
        if (!wire.get_standard()) {
            continue;
        }
        auto wireStandard = wire.get_standard().value();
        if (std::find(wireStandards.begin(), wireStandards.end(), wireStandard) == wireStandards.end()) {
            wireStandards.push_back(wireStandard);
        }
    }
    std::vector<std::string> wireStandardsString;
    for (auto wireStandard : wireStandards) {
        json wireStandardString;        
        to_json(wireStandardString, wireStandard);
        wireStandardsString.push_back(wireStandardString);
    }
    return wireStandardsString;
}

std::string calculate_gap_reluctance(std::string coreGapData, std::string modelNameString){
    try {
        OpenMagnetics::ReluctanceModels modelName;
        from_json(modelNameString, modelName);
        auto reluctanceModel = OpenMagnetics::ReluctanceModel::factory(modelName);

        CoreGap coreGap(json::parse(coreGapData));

        auto coreGapResult = reluctanceModel->get_gap_reluctance(coreGap);
        json result;
        to_json(result, coreGapResult);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_gap_reluctance_model_information(){
    try {
        json info;
        info["information"] = OpenMagnetics::ReluctanceModel::get_models_information();
        info["errors"] = OpenMagnetics::ReluctanceModel::get_models_errors();
        info["internal_links"] = OpenMagnetics::ReluctanceModel::get_models_internal_links();
        info["external_links"] = OpenMagnetics::ReluctanceModel::get_models_external_links();
        return info.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

double calculate_inductance_from_number_turns_and_gapping(std::string coreData,
                                                          std::string coilData,
                                                          std::string operatingPointData,
                                                          std::string modelsData){
    try {
        json coreJson = json::parse(coreData);
        MAS::CoreFunctionalDescription coreFunctionalDescription(coreJson["functionalDescription"]);
        MAS::CoreProcessedDescription coreProcessedDescription(coreJson["processedDescription"]);
        OpenMagnetics::Core core;
        core.set_functional_description(coreFunctionalDescription);
        core.set_processed_description(coreProcessedDescription);
        // geometricalDescription is optional; skip when missing/null to avoid the
        // nlohmann::json "type must be array, but is null" constructor error.
        if (coreJson.contains("geometricalDescription") && !coreJson["geometricalDescription"].is_null()) {
            std::vector<MAS::CoreGeometricalDescriptionElement> coreGeometricalDescription(coreJson["geometricalDescription"]);
            core.set_geometrical_description(coreGeometricalDescription);
        }
        OpenMagnetics::Coil coil(json::parse(coilData), false);

        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();

        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::MagnetizingInductance magnetizingInductanceObj(reluctanceModelName);
        double magnetizingInductance;
        if (operatingPointData != "") {
            OperatingPoint operatingPoint(json::parse(operatingPointData));
            magnetizingInductance = magnetizingInductanceObj.calculate_inductance_from_number_turns_and_gapping(core, coil, &operatingPoint).get_magnetizing_inductance().get_nominal().value();
        }
        else {
            magnetizingInductance = magnetizingInductanceObj.calculate_inductance_from_number_turns_and_gapping(core, coil).get_magnetizing_inductance().get_nominal().value();
        }

        return magnetizingInductance;
    }
    catch (const std::exception &exc) {
        std::cerr << coreData << std::endl;
        std::cerr << coilData << std::endl;
        std::cerr << operatingPointData << std::endl;
        std::cerr << modelsData << std::endl;
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
        return -1;
    }
}


// Canonical coil-aware form (preferred): turns are sized against the actual
// winding via MKF's coil-taking overload.
double calculate_number_turns_from_gapping_and_inductance(std::string coreData,
                                                          std::string coilData,
                                                          std::string inputsData,
                                                          std::string modelsData){
    try {
        OpenMagnetics::Core core(json::parse(coreData));
        OpenMagnetics::Coil coil(json::parse(coilData), false);
        OpenMagnetics::Inputs inputs(json::parse(inputsData));

        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();

        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::MagnetizingInductance magnetizingInductanceObj(reluctanceModelName);
        double numberTurns = magnetizingInductanceObj.calculate_number_turns_from_gapping_and_inductance(core, coil, &inputs);

        return numberTurns;
    }
    catch (const std::exception &exc) {
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
        return -1;
    }
}

// Legacy 3-argument form (no coilData) — backward-compat backup. Forwards to
// MKF's 3-argument overload, which synthesizes a single-primary-winding coil.
double calculate_number_turns_from_gapping_and_inductance_legacy(std::string coreData,
                                                                 std::string inputsData,
                                                                 std::string modelsData){
    try {
        OpenMagnetics::Core core(json::parse(coreData));
        OpenMagnetics::Inputs inputs(json::parse(inputsData));

        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();

        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::MagnetizingInductance magnetizingInductanceObj(reluctanceModelName);
        double numberTurns = magnetizingInductanceObj.calculate_number_turns_from_gapping_and_inductance(core, &inputs);

        return numberTurns;
    }
    catch (const std::exception &exc) {
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
        return -1;
    }
}


std::string calculate_gapping_from_number_turns_and_inductance(std::string coreData,
                                                               std::string coilData,
                                                               std::string inputsData,
                                                               std::string gappingTypeString,
                                                               int decimals,
                                                               std::string modelsData){
    try {
        OpenMagnetics::Core core(json::parse(coreData));
        OpenMagnetics::Coil coil(json::parse(coilData), false);
        OpenMagnetics::Inputs inputs(json::parse(inputsData));

        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        OpenMagnetics::GappingType gappingType;
        OpenMagnetics::from_json(gappingTypeString, gappingType);
        
        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::MagnetizingInductance magnetizingInductanceObj(reluctanceModelName);
        std::vector<CoreGap> gapping = magnetizingInductanceObj.calculate_gapping_from_number_turns_and_inductance(core,
                                                                                                           coil,
                                                                                                           &inputs,
                                                                                                           gappingType,
                                                                                                           decimals);

        core.set_processed_description(std::nullopt);
        core.set_geometrical_description(std::nullopt);
        core.get_mutable_functional_description().set_gapping(gapping);
        core.process_data();
        core.process_gap();
        auto geometricalDescription = core.create_geometrical_description();
        core.set_geometrical_description(geometricalDescription);

        json result;
        to_json(result, core);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_core_losses(std::string coreData,
                                  std::string coilData,
                                  std::string inputsData,    
                                  std::string modelsData,    
                                  int operatingPointIndex){
    try {
        json coreJson = json::parse(coreData);
        MAS::CoreFunctionalDescription coreFunctionalDescription(coreJson["functionalDescription"]);
        MAS::CoreProcessedDescription coreProcessedDescription(coreJson["processedDescription"]);
        OpenMagnetics::Core core;
        core.set_functional_description(coreFunctionalDescription);
        core.set_processed_description(coreProcessedDescription);
        // geometricalDescription is optional; skip when missing/null to avoid the
        // nlohmann::json "type must be array, but is null" constructor error.
        if (coreJson.contains("geometricalDescription") && !coreJson["geometricalDescription"].is_null()) {
            std::vector<MAS::CoreGeometricalDescriptionElement> coreGeometricalDescription(coreJson["geometricalDescription"]);
            core.set_geometrical_description(coreGeometricalDescription);
        }
        OpenMagnetics::Coil coil(json::parse(coilData), false);

        OpenMagnetics::MagnetizingInductance magnetizingInductanceModel;
        double magnetizingInductance = magnetizingInductanceModel.calculate_inductance_from_number_turns_and_gapping(core, coil).get_magnetizing_inductance().get_nominal().value();

        OpenMagnetics::Inputs inputs(json::parse(inputsData), true, magnetizingInductance);
        auto operatingPoint = inputs.get_operating_point(operatingPointIndex);
        OperatingPointExcitation excitation = operatingPoint.get_excitations_per_winding()[0];
        // double magnetizingInductance = OpenMagnetics::resolve_dimensional_values(inputs.get_design_requirements().get_magnetizing_inductance());
        if (!excitation.get_current()) {
            auto magnetizingCurrent = OpenMagnetics::Inputs::calculate_magnetizing_current(excitation, magnetizingInductance, true, 0.0);
            excitation.set_current(magnetizingCurrent);
            operatingPoint.get_mutable_excitations_per_winding()[0] = excitation;
        }

        auto defaults = OpenMagnetics::Defaults();

        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();

        auto reluctanceModelName = OpenMagnetics::defaults.reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }
        auto coreLossesModelName = OpenMagnetics::defaults.coreLossesModelDefault;
        if (models.find("coreLosses") != models.end()) {
            OpenMagnetics::from_json(models["coreLosses"], coreLossesModelName);
        }
        auto coreTemperatureModelName = OpenMagnetics::defaults.coreTemperatureModelDefault;
        if (models.find("coreTemperature") != models.end()) {
            OpenMagnetics::from_json(models["coreTemperature"], coreTemperatureModelName);
        }

        OpenMagnetics::Magnetic magnetic;
        magnetic.set_core(core);
        magnetic.set_coil(coil);

        OpenMagnetics::MagneticSimulator magneticSimulator;
        magneticSimulator.set_core_losses_model_name(coreLossesModelName);
        // The core-temperature-model knob moved from MagneticSimulator to Settings
        // (MKF commit 10aa82c: simulator now derives core-loss temperature via the
        // thermal-network model). Set it on Settings to honor the request payload.
        OpenMagnetics::Settings::GetInstance().set_core_temperature_model(coreTemperatureModelName);
        magneticSimulator.set_reluctance_model_name(reluctanceModelName);
        auto coreLossesOutput = magneticSimulator.calculate_core_losses(operatingPoint, magnetic);
        json result;
        to_json(result, coreLossesOutput);

        OpenMagnetics::MagnetizingInductance magnetizingInductanceObj(reluctanceModelName);
        auto magneticFluxDensity = magnetizingInductanceObj.calculate_inductance_and_magnetic_flux_density(core, coil, &operatingPoint).second;
        excitation.set_magnetic_flux_density(magneticFluxDensity);

        result["magneticFluxDensityPeak"] = magneticFluxDensity.get_processed().value().get_peak().value();

        double frequency = OpenMagnetics::Inputs::get_switching_frequency(excitation);
        double magneticFluxDensityAcPeakToPeak = OpenMagnetics::Inputs::get_magnetic_flux_density_peak_to_peak(excitation, frequency);
        result["magneticFluxDensityAcPeak"] = magneticFluxDensityAcPeakToPeak / 2;
        result["voltageRms"] = operatingPoint.get_mutable_excitations_per_winding()[0].get_voltage().value().get_processed().value().get_rms().value();
        result["currentRms"] = operatingPoint.get_mutable_excitations_per_winding()[0].get_current().value().get_processed().value().get_rms().value();
        result["apparentPower"] = operatingPoint.get_mutable_excitations_per_winding()[0].get_voltage().value().get_processed().value().get_rms().value() * operatingPoint.get_mutable_excitations_per_winding()[0].get_current().value().get_processed().value().get_rms().value();
        double coreTemperature = coreLossesOutput.get_temperature();
        result["maximumCoreTemperature"] = coreTemperature;
        result["maximumCoreTemperatureRise"] = coreTemperature - operatingPoint.get_conditions().get_ambient_temperature();
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << coreData << std::endl;
        std::cerr << coilData << std::endl;
        std::cerr << inputsData << std::endl;
        std::cerr << modelsData << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_core_losses_model_information(std::string material){
    try {
        json info;
        info["information"] = OpenMagnetics::CoreLossesModel::get_models_information();
        info["errors"] = OpenMagnetics::CoreLossesModel::get_models_errors();
        info["internal_links"] = OpenMagnetics::CoreLossesModel::get_models_internal_links();
        info["external_links"] = OpenMagnetics::CoreLossesModel::get_models_external_links();
        info["available_models"] = OpenMagnetics::CoreLossesModel::get_methods_string(material);
        return info.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_core_temperature_model_information(){
    try {
        json info;
        info["information"] = OpenMagnetics::CoreTemperatureModel::get_models_information();
        info["errors"] = OpenMagnetics::CoreTemperatureModel::get_models_errors();
        info["internal_links"] = OpenMagnetics::CoreTemperatureModel::get_models_internal_links();
        info["external_links"] = OpenMagnetics::CoreTemperatureModel::get_models_external_links();
        return info.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_induced_voltage(std::string excitationString, double magnetizingInductance){
    try {
        OperatingPointExcitation excitation(json::parse(excitationString));

        auto voltage = OpenMagnetics::Inputs::calculate_induced_voltage(excitation, magnetizingInductance);

        json result;
        to_json(result, voltage);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_induced_current(std::string excitationString, double magnetizingInductance){
    try {
        OperatingPointExcitation excitation(json::parse(excitationString));

        auto current = OpenMagnetics::Inputs::calculate_magnetizing_current(excitation, magnetizingInductance, true, 0.0);

        if (excitation.get_voltage()) {
            if (excitation.get_voltage().value().get_processed()) {
                if (excitation.get_voltage().value().get_processed().value().get_duty_cycle()) {
                    auto processed = current.get_processed().value();
                    processed.set_duty_cycle(excitation.get_voltage().value().get_processed().value().get_duty_cycle().value());
                    current.set_processed(processed);
                }
            }
        }

        json result;
        to_json(result, current);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_reflected_secondary(std::string primaryExcitationString, double turnRatio){
    try {
        OperatingPointExcitation primaryExcitation(json::parse(primaryExcitationString));

        OperatingPointExcitation excitationOfThisWinding(primaryExcitation);
        auto currentSignalDescriptorProcessed = OpenMagnetics::Inputs::calculate_basic_processed_data(primaryExcitation.get_current().value().get_waveform().value());
        auto voltageSignalDescriptorProcessed = OpenMagnetics::Inputs::calculate_basic_processed_data(primaryExcitation.get_voltage().value().get_waveform().value());

        auto voltageSignalDescriptor = OpenMagnetics::Inputs::reflect_waveform(primaryExcitation.get_voltage().value(), 1.0 / turnRatio, voltageSignalDescriptorProcessed.get_label());
        auto currentSignalDescriptor = OpenMagnetics::Inputs::reflect_waveform(primaryExcitation.get_current().value(), turnRatio, currentSignalDescriptorProcessed.get_label());

        auto voltageSampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(voltageSignalDescriptor.get_waveform().value(), excitationOfThisWinding.get_frequency());
        voltageSignalDescriptor.set_harmonics(OpenMagnetics::Inputs::calculate_harmonics_data(voltageSampledWaveform, excitationOfThisWinding.get_frequency()));
        voltageSignalDescriptor.set_processed(OpenMagnetics::Inputs::calculate_processed_data(voltageSignalDescriptor, voltageSampledWaveform, true));

        auto currentSampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(currentSignalDescriptor.get_waveform().value(), excitationOfThisWinding.get_frequency());
        currentSignalDescriptor.set_harmonics(OpenMagnetics::Inputs::calculate_harmonics_data(currentSampledWaveform, excitationOfThisWinding.get_frequency()));
        currentSignalDescriptor.set_processed(OpenMagnetics::Inputs::calculate_processed_data(currentSignalDescriptor, currentSampledWaveform, true));

        excitationOfThisWinding.set_voltage(voltageSignalDescriptor);
        excitationOfThisWinding.set_current(currentSignalDescriptor);

        json result;
        to_json(result, excitationOfThisWinding);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_reflected_primary(std::string secondaryExcitationString, double turnRatio){
    try {
        OperatingPointExcitation secondaryExcitation(json::parse(secondaryExcitationString));

        OperatingPointExcitation excitationOfThisWinding(secondaryExcitation);
        auto voltageSignalDescriptor = OpenMagnetics::Inputs::reflect_waveform(secondaryExcitation.get_voltage().value(), turnRatio);
        auto currentSignalDescriptor = OpenMagnetics::Inputs::reflect_waveform(secondaryExcitation.get_current().value(), 1.0 / turnRatio);

        auto voltageSampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(voltageSignalDescriptor.get_waveform().value(), excitationOfThisWinding.get_frequency());
        voltageSignalDescriptor.set_harmonics(OpenMagnetics::Inputs::calculate_harmonics_data(voltageSampledWaveform, excitationOfThisWinding.get_frequency()));
        voltageSignalDescriptor.set_processed(OpenMagnetics::Inputs::calculate_processed_data(voltageSignalDescriptor, voltageSampledWaveform, true));

        auto currentSampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(currentSignalDescriptor.get_waveform().value(), excitationOfThisWinding.get_frequency());
        currentSignalDescriptor.set_harmonics(OpenMagnetics::Inputs::calculate_harmonics_data(currentSampledWaveform, excitationOfThisWinding.get_frequency()));
        currentSignalDescriptor.set_processed(OpenMagnetics::Inputs::calculate_processed_data(currentSignalDescriptor, currentSampledWaveform, true));

        excitationOfThisWinding.set_voltage(voltageSignalDescriptor);
        excitationOfThisWinding.set_current(currentSignalDescriptor);

        json result;
        to_json(result, excitationOfThisWinding);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

double calculate_instantaneous_power(std::string excitationString){
    OperatingPointExcitation excitation(json::parse(excitationString));

    // ABT #223: this used to "guard" with
    //     if (!excitation.get_current().value().get_processed().value().get_rms().value())
    // which dereferences the very optionals it is meant to test — on the DEFAULT
    // manual operating-point view the excitations carry a waveform but no
    // `processed` block yet, so the guard itself threw bad_optional_access and the
    // worker reported a generic "Error calling calculate_instantaneous_power".
    // The processed data was never needed: Inputs::calculate_instantaneous_power
    // works from the voltage/current WAVEFORMS and already throws specific
    // MISSING_DATA messages when one is absent.
    return OpenMagnetics::Inputs::calculate_instantaneous_power(excitation);
}

double calculate_rms_power(std::string excitationString){
    try {
        OperatingPointExcitation excitation(json::parse(excitationString));

        if (!excitation.get_voltage() || !excitation.get_current()) {
            return 0.0;
        }

        auto voltageSignalDescriptor = excitation.get_voltage().value();
        auto currentSignalDescriptor = excitation.get_current().value();

        if (!voltageSignalDescriptor.get_waveform() || !currentSignalDescriptor.get_waveform()) {
            return 0.0;
        }

        if (!voltageSignalDescriptor.get_processed() || !voltageSignalDescriptor.get_processed()->get_rms()) {
            auto voltageSampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(voltageSignalDescriptor.get_waveform().value(), excitation.get_frequency());
            voltageSignalDescriptor.set_harmonics(OpenMagnetics::Inputs::calculate_harmonics_data(voltageSampledWaveform, excitation.get_frequency()));
            voltageSignalDescriptor.set_processed(OpenMagnetics::Inputs::calculate_processed_data(voltageSignalDescriptor, voltageSampledWaveform, true));
        }

        if (!currentSignalDescriptor.get_processed() || !currentSignalDescriptor.get_processed()->get_rms()) {
            auto currentSampledWaveform = OpenMagnetics::Inputs::calculate_sampled_waveform(currentSignalDescriptor.get_waveform().value(), excitation.get_frequency());
            currentSignalDescriptor.set_harmonics(OpenMagnetics::Inputs::calculate_harmonics_data(currentSampledWaveform, excitation.get_frequency()));
            currentSignalDescriptor.set_processed(OpenMagnetics::Inputs::calculate_processed_data(currentSignalDescriptor, currentSampledWaveform, true));
        }

        if (!currentSignalDescriptor.get_processed()->get_rms() || !voltageSignalDescriptor.get_processed()->get_rms()) {
            return 0.0;
        }

        double rmsPower = currentSignalDescriptor.get_processed().value().get_rms().value() * voltageSignalDescriptor.get_processed().value().get_rms().value();

        return rmsPower;
    }
    catch (const std::exception &exc) {
        return 0.0;
    }
}

double resolve_dimension_with_tolerance(std::string dimensionWithToleranceString) {
    DimensionWithTolerance dimensionWithTolerance(json::parse(dimensionWithToleranceString));
    return OpenMagnetics::resolve_dimensional_values(dimensionWithTolerance);
}

std::string calculate_basic_processed_data(std::string waveformString) {
    try {
        Waveform waveform(json::parse(waveformString));
        auto processed = OpenMagnetics::Inputs::calculate_basic_processed_data(waveform);
        json result;
        to_json(result, processed);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string create_waveform(std::string processedString, double frequency) {
    try {
        ProcessedWaveform processed(json::parse(processedString));
        auto waveform = OpenMagnetics::Inputs::create_waveform(processed, frequency);
        json result;
        to_json(result, waveform);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string scale_waveform_time_to_frequency(std::string waveformString, double newFrequency) {
    try {
        Waveform waveform(json::parse(waveformString));
        auto scaledWaveform = OpenMagnetics::Inputs::scale_time_to_frequency(waveform, newFrequency);
        json result;
        to_json(result, scaledWaveform);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string scale_excitation_time_to_frequency(std::string excitationString, double newFrequency) {
    try {
        OperatingPointExcitation excitation(json::parse(excitationString));
        OpenMagnetics::Inputs::scale_time_to_frequency(excitation, newFrequency, false, true);
        json result;
        to_json(result, excitation);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_insulation(std::string inputsString){
    json result;
    result["creepageDistance"] = 0.0;
    result["clearance"] = 0.0;
    result["withstandVoltage"] = 0.0;
    result["distanceThroughInsulation"] = 0.0;
    result["errorMessage"] = "";
    try {
        // Build Inputs by hand instead of via OpenMagnetics::Inputs(json,
        // false) — that constructor calls check_integrity(), which tries
        // to synthesise a magnetizing current from the voltage waveform
        // and throws when prerequisites it doesn't actually need for the
        // insulation calc (waveform, magnetizing inductance, etc.) are
        // missing. The four insulation calculators only read the voltage
        // *processed* peak/rms, the altitude, and the insulation
        // requirements — none of which require a magnetizing-current pass.
        auto j = json::parse(inputsString);
        OpenMagnetics::compat::migrate_pre_1_0(j);
        OpenMagnetics::Inputs inputs;
        from_json(j, inputs);

        auto standard = OpenMagnetics::InsulationCoordinator();
        result["creepageDistance"] = standard.calculate_creepage_distance(inputs);
        result["clearance"] = standard.calculate_clearance(inputs);
        result["withstandVoltage"] = standard.calculate_withstand_voltage(inputs);
        result["distanceThroughInsulation"] = standard.calculate_distance_through_insulation(inputs);
    }
    catch(const std::runtime_error& re)
    {
        result["errorMessage"] = std::string{re.what()};
    }
    catch(const std::exception& ex)
    {
        result["errorMessage"] = std::string{ex.what()};
    }
    catch(...)
    {
        result["errorMessage"] = "Unknown failure occurred. Possible memory corruption";
    }
    return result.dump(4);
}

// All three CSV-import bindings return their result as a JSON string on
// success, or a string beginning with "ERROR:" on failure. The frontend
// proxy checks the prefix before JSON.parse and surfaces the real reason
// (file:line:column, missing time column, etc.) to the user instead of
// the previous catch-all "please check column names and frequency".
std::string extract_operating_point(std::string fileString, size_t numberWindings, double frequency, double desiredMagnetizingInductance, std::string mapColumnNamesString){
    try {
        std::vector<std::map<std::string, std::string>> mapColumnNames =
            json::parse(mapColumnNamesString).get<std::vector<std::map<std::string, std::string>>>();
        auto reader = OpenMagnetics::CircuitSimulationReader(fileString, true);
        auto operatingPoint = reader.extract_operating_point(numberWindings, frequency, mapColumnNames);
        operatingPoint = OpenMagnetics::Inputs::process_operating_point(operatingPoint, desiredMagnetizingInductance);
        json result;
        to_json(result, operatingPoint);
        return result.dump(4);
    }
    catch (const std::exception& e) {
        return std::string("ERROR:") + e.what();
    }
    catch (...) {
        return std::string("ERROR:unknown failure while extracting operating point from circuit simulation");
    }
}

std::string extract_map_column_names(std::string fileString, size_t numberWindings, double frequency){
    try {
        auto reader = OpenMagnetics::CircuitSimulationReader(fileString, true);
        auto columnNames = reader.extract_map_column_names(numberWindings, frequency);

        json result = json::array();
        for (auto& columnName : columnNames) {
            json aux;
            for (auto& [signal, name] : columnName) {
                aux[signal] = name;
            }
            result.push_back(aux);
        }

        return result.dump(4);
    }
    catch (const std::exception& e) {
        return std::string("ERROR:") + e.what();
    }
    catch (...) {
        return std::string("ERROR:unknown failure while extracting column-name mapping");
    }
}

std::string extract_column_names(std::string fileString){
    try {
        auto reader = OpenMagnetics::CircuitSimulationReader(fileString, true);
        auto columnNames = reader.extract_column_names();

        json result = json::array();
        for (auto& columnName : columnNames) {
            result.push_back(columnName);
        }

        return result.dump(4);
    }
    catch (const std::exception& e) {
        return std::string("ERROR:") + e.what();
    }
    catch (...) {
        return std::string("ERROR:unknown failure while extracting column names");
    }
}

std::vector<int> calculate_number_turns(int numberTurnsPrimary, std::string designRequirementsString){
    DesignRequirements designRequirements(json::parse(designRequirementsString));

    OpenMagnetics::NumberTurns numberTurns(numberTurnsPrimary, designRequirements);
    auto numberTurnsCombination = numberTurns.get_next_number_turns_combination();

    std::vector<int> numberTurnsResult;
    for (auto turns : numberTurnsCombination) {
        numberTurnsResult.push_back(static_cast<std::make_signed<int>::type>(turns));
    }
    return numberTurnsResult;
}

double calculate_dc_resistance_per_meter(std::string wireString, double temperature){
    try {
        auto wire = resolve_wire_from_string(wireString);
        auto dcResistancePerMeter = OpenMagnetics::WindingOhmicLosses::calculate_dc_resistance_per_meter(wire, temperature);
        return dcResistancePerMeter;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error in calculate_dc_resistance_per_meter: " << ex.what() << std::endl;
        return -1;
    }
}

std::vector<double> calculate_dc_resistance_per_winding(std::string coilString, double temperature){
    OpenMagnetics::Coil coil(json::parse(coilString), false);
    auto dcResistancePerWinding = OpenMagnetics::WindingOhmicLosses::calculate_dc_resistance_per_winding(coil, temperature);
    return dcResistancePerWinding;
}

double calculate_dc_losses_per_meter(std::string wireString, std::string currentString, double temperature){
    try {
        auto wire = resolve_wire_from_string(wireString);
        SignalDescriptor current(json::parse(currentString));
        
        // Ensure current has processed data
        if (!ensure_current_processed(current, "calculate_dc_losses_per_meter")) {
            return -1;
        }
        
        auto dcLossesPerMeter = OpenMagnetics::WindingOhmicLosses::calculate_ohmic_losses_per_meter(wire, current, temperature);
        return dcLossesPerMeter;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error in calculate_dc_losses_per_meter: " << ex.what() << std::endl;
        return -1;
    }
}

double calculate_skin_ac_factor(std::string wireString, std::string currentString, double temperature){
    try {
        auto wire = resolve_wire_from_string(wireString);
        SignalDescriptor current(json::parse(currentString));
        
        // Ensure current has processed data
        if (!ensure_current_processed(current, "calculate_skin_ac_factor")) {
            return -1;
        }
        
        auto dcLossesPerMeter = OpenMagnetics::WindingOhmicLosses::calculate_ohmic_losses_per_meter(wire, current, temperature);
        auto [skinLossesPerMeter, _] = OpenMagnetics::WindingSkinEffectLosses::calculate_skin_effect_losses_per_meter(wire, current, temperature);
        auto skinAcFactor = (skinLossesPerMeter + dcLossesPerMeter) / dcLossesPerMeter;
        return skinAcFactor;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error in calculate_skin_ac_factor: " << ex.what() << std::endl;
        return -1;
    }
}

double calculate_skin_ac_losses_per_meter(std::string wireString, std::string currentString, double temperature){
    try {
        auto wire = resolve_wire_from_string(wireString);
        SignalDescriptor current(json::parse(currentString));
        
        // Ensure current has processed data
        if (!ensure_current_processed(current, "calculate_skin_ac_losses_per_meter")) {
            return -1;
        }
        
        auto [skinLossesPerMeter, _] = OpenMagnetics::WindingSkinEffectLosses::calculate_skin_effect_losses_per_meter(wire, current, temperature);
        return skinLossesPerMeter;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error in calculate_skin_ac_losses_per_meter: " << ex.what() << std::endl;
        return -1;
    }
}

double calculate_skin_ac_resistance_per_meter(std::string wireString, std::string currentString, double temperature){
    try {
        auto wire = resolve_wire_from_string(wireString);
        SignalDescriptor current(json::parse(currentString));
        
        // Ensure current has processed data
        if (!ensure_current_processed(current, "calculate_skin_ac_resistance_per_meter")) {
            return -1;
        }
        
        auto dcLossesPerMeter = OpenMagnetics::WindingOhmicLosses::calculate_ohmic_losses_per_meter(wire, current, temperature);
        auto [skinLossesPerMeter, _] = OpenMagnetics::WindingSkinEffectLosses::calculate_skin_effect_losses_per_meter(wire, current, temperature);
        auto skinAcFactor = (skinLossesPerMeter + dcLossesPerMeter) / dcLossesPerMeter;
        auto dcResistancePerMeter = OpenMagnetics::WindingOhmicLosses::calculate_dc_resistance_per_meter(wire, temperature);

        return dcResistancePerMeter * skinAcFactor;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error in calculate_skin_ac_resistance_per_meter: " << ex.what() << std::endl;
        return -1;
    }
}

double calculate_effective_current_density(std::string wireString, std::string currentString, double temperature){
    try {
        auto wire = resolve_wire_from_string(wireString);
        SignalDescriptor current(json::parse(currentString));
        
        // Ensure current has processed data
        if (!ensure_current_processed(current, "calculate_effective_current_density")) {
            return -1;
        }
        
        auto effectiveCurrentDensity = wire.calculate_effective_current_density(current, temperature);
        return effectiveCurrentDensity;
    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error in calculate_effective_current_density: " << ex.what() << std::endl;
        return -1;
    }
}

double calculate_effective_skin_depth(std::string material, std::string currentString, double temperature){
    try {
        SignalDescriptor current(json::parse(currentString));

        // Ensure current has processed data with effective frequency
        if (!ensure_current_processed_for_effective_frequency(current, "calculate_effective_skin_depth")) {
            return -1;
        }

        auto currentEffectiveFrequency = current.get_processed()->get_effective_frequency().value();
        double effectiveSkinDepth = OpenMagnetics::WindingSkinEffectLosses::calculate_skin_depth(material, currentEffectiveFrequency, temperature);
        return effectiveSkinDepth;
    }
    catch(const std::exception& ex)
    {
        return -1;
    }
}

std::vector<std::string> get_available_winding_orientations(){
    std::vector<std::string> orientations;

    for (size_t index = 0; index < magic_enum::enum_count<WindingOrientation>(); ++index) {
        auto orientation = static_cast<WindingOrientation>(index);
        orientations.push_back(OpenMagnetics::to_string(orientation));
    }
    return orientations;
}

std::vector<std::string> get_available_coil_alignments(){
    std::vector<std::string> coilAlignments;

    for (size_t index = 0; index < magic_enum::enum_count<CoilAlignment>(); ++index) {
        auto coilAlignment = static_cast<CoilAlignment>(index);
        coilAlignments.push_back(OpenMagnetics::to_string(coilAlignment));
    }
    return coilAlignments;
}

bool check_requirement(std::string requirementString, double value){
    try {
        DimensionWithTolerance requirement(json::parse(requirementString));
        bool result = OpenMagnetics::check_requirement(requirement, value);
        return result;
    }
    catch(const std::exception& ex)
    {
        return false;
    }
}

void process_coil_configuration(OpenMagnetics::Coil& coil, json configuration, std::optional<size_t> repetitions = std::nullopt, std::optional<std::vector<double>> proportionPerWinding = std::nullopt, std::optional<std::vector<size_t>> pattern = std::nullopt) {
    if (repetitions && proportionPerWinding && pattern) {
        if (configuration["_layersOrientation"].is_object()) {
            auto layersOrientationPerSection = std::map<std::string, WindingOrientation>(configuration["_layersOrientation"]);
            for (auto [sectionName, layerOrientation] : layersOrientationPerSection) {
                coil.set_layers_orientation(layerOrientation, sectionName);
            }
        }
        else if (configuration["_layersOrientation"].is_array()) {
            coil.wind_by_sections(proportionPerWinding.value(), pattern.value(), repetitions.value());
            if (coil.get_sections_description()) {
                auto sections = coil.get_sections_description_conduction();
                auto layersOrientationPerSection = std::vector<WindingOrientation>(configuration["_layersOrientation"]);
                for (size_t sectionIndex = 0; sectionIndex < sections.size(); ++sectionIndex) {
                    if (sectionIndex < layersOrientationPerSection.size()) {
                        coil.set_layers_orientation(layersOrientationPerSection[sectionIndex], sections[sectionIndex].get_name());
                    }
                }
            }
        }
        else {
            WindingOrientation layerOrientation(configuration["_layersOrientation"]);
            coil.set_layers_orientation(layerOrientation);

        }
        if (configuration["_turnsAlignment"].is_object()) {
            auto turnsAlignmentPerSection = std::map<std::string, CoilAlignment>(configuration["_turnsAlignment"]);
            for (auto [sectionName, turnsAlignment] : turnsAlignmentPerSection) {
                coil.set_turns_alignment(turnsAlignment, sectionName);
            }
        }
        else if (configuration["_turnsAlignment"].is_array()) {
            coil.wind_by_sections(proportionPerWinding.value(), pattern.value(), repetitions.value());
            if (coil.get_sections_description()) {
                auto sections = coil.get_sections_description_conduction();
                auto turnsAlignmentPerSection = std::vector<CoilAlignment>(configuration["_turnsAlignment"]);
                for (size_t sectionIndex = 0; sectionIndex < sections.size(); ++sectionIndex) {
                    if (sectionIndex < turnsAlignmentPerSection.size()) {
                        coil.set_turns_alignment(turnsAlignmentPerSection[sectionIndex], sections[sectionIndex].get_name());
                    }
                }
            }
        }
        else {
            CoilAlignment turnsAlignment(configuration["_turnsAlignment"]);
            coil.set_turns_alignment(turnsAlignment);
        }
    }
    else {
        if (configuration.contains("_layersOrientation")) {
            coil.set_layers_orientation(configuration["_layersOrientation"]);
        }
        if (configuration.contains("_turnsAlignment")) {
            coil.set_turns_alignment(configuration["_turnsAlignment"]);
        }
    }

    // Winding-style overrides (winding studio): winding name → forced
    // consecutive-parallels (multifilar bundle) / consecutive-turns.
    if (configuration.contains("_windingStyle") && configuration["_windingStyle"].is_object()) {
        auto windingStyleOverrides = std::map<std::string, WindingStyle>(configuration["_windingStyle"]);
        coil.preload_winding_style_overrides(windingStyleOverrides);
    }

    if (configuration.contains("_interleavingLevel")) {
        coil.set_interleaving_level(configuration["_interleavingLevel"]);
    }
    if (configuration.contains("_windingOrientation")) {
        coil.set_winding_orientation(configuration["_windingOrientation"]);
    }
    if (configuration.contains("_sectionAlignment")) {
        coil.set_section_alignment(configuration["_sectionAlignment"]);
    }
    if (configuration.contains("_sectionAlignment")) {
        coil.set_section_alignment(configuration["_sectionAlignment"]);
    }

    // Shielding requirements must be set before any insulation configuration, so the
    // insulation setters below can splice the shield layers into the interface stacks
    if (configuration.contains("_shielding")) {
        auto shieldingRequirements = std::vector<MAS::ShieldingRequirement>(configuration["_shielding"]);
        coil.set_shielding_requirements(shieldingRequirements);
    }

    // NOTE: zero thickness is intentionally allowed. A zero-thickness inter-layer
    // insulation layer acts as a geometric/thermal "marker" that lets adjacent
    // layer turns participate in the thermal graph (R_layer = 0, only wire enamel
    // contributes resistance). Coil::set_interlayer_insulation always assigns a
    // default material when none is provided, so Temperature.cpp's no-material
    // path is never hit.
    if (configuration.contains("_interlayerInsulationThickness")) {
        coil.set_interlayer_insulation(configuration["_interlayerInsulationThickness"], std::nullopt, std::nullopt, false);
    }
    if (configuration.contains("_intersectionInsulationThickness")) {
        coil.set_intersection_insulation(configuration["_intersectionInsulationThickness"], 1, std::nullopt, std::nullopt, false);
    }

}

static std::string wind_impl(const std::string& coilString, const std::string& coreColumnsString, size_t repetitions, const std::string& proportionPerWindingString, const std::string& patternString, const std::string& marginPairsString, const std::string& customSectionRectsString = "", bool delimitAndCompact = true) {
    try {
        auto coilJson = json::parse(coilString);
        auto marginPairs = std::vector<std::vector<double>>(json::parse(marginPairsString));

        OpenMagnetics::Settings::GetInstance().set_coil_wind_even_if_not_fit(true);
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(delimitAndCompact);
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(true);

        std::vector<double> proportionPerWinding = json::parse(proportionPerWindingString);
        std::vector<size_t> pattern = json::parse(patternString);
        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        OpenMagnetics::Coil coil;
        coil.set_bobbin_from_json(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.preload_margins(marginPairs);
        if (!coreColumnsString.empty()) {
            // Multi-column placement: windings/sections placed in non-main winding
            // windows need the core columns to build their lateral wound-column
            // frames (Coil throws a specific error without them).
            std::vector<ColumnElement> coreColumns = json::parse(coreColumnsString);
            coil.set_core_columns(coreColumns);
        }
        if (!customSectionRectsString.empty()) {
            // Hand-drawn section rectangles (winding studio): re-imposed at the
            // end of the wind, after compaction — a drawn section never moves.
            json rectsJson = json::parse(customSectionRectsString);
            std::map<std::string, std::pair<std::vector<double>, std::vector<double>>> customSectionRects;
            for (auto& [sectionName, rect] : rectsJson.items()) {
                customSectionRects[sectionName] = {
                    rect.at("coordinates").get<std::vector<double>>(),
                    rect.at("dimensions").get<std::vector<double>>()};
            }
            coil.preload_custom_section_rects(customSectionRects);
        }

        process_coil_configuration(coil, coilJson, repetitions, proportionPerWinding, pattern);

        if (proportionPerWinding.size() == winding.size()) {
            if (pattern.size() > 0 && repetitions > 0) {
                coil.wind(proportionPerWinding, pattern, repetitions);
            }
            else if (repetitions > 0) {
                coil.wind(repetitions);
            }
            else {
                coil.wind();
            }
        }
        else {
            if (pattern.size() > 0 && repetitions > 0) {
                coil.wind(pattern, repetitions);
            }
            else if (repetitions > 0) {
                coil.wind(repetitions);
            }
            else {
                coil.wind();
            }
        }

        if (!coil.get_turns_description()) {
            {
                // ABT #930: name why the wind produced nothing — a bare "Turns not created"
                // reads as a broken winder even when MKF is right to refuse the geometry.
                const auto& fitReason = coil.get_last_fit_failure();
                throw std::runtime_error(fitReason.empty()
                    ? std::string("Turns not created: the winding does not fit its window, and the "
                                  "reason could not be narrowed further.")
                    : "Turns not created. " + fitReason);
            }
        }

        if (delimitAndCompact) {
            // Explicitly call delimit_and_compact to ensure toroidal additional turns are compacted
            coil.delimit_and_compact();
            // ...and re-impose the drawn rectangles it may have moved (no-op
            // when none were preloaded).
            coil.apply_custom_section_rects();
        }

        json result;
        to_json(result, coil);

        // Debug: Check if additional_coordinates are in the output
        size_t turnsWithAdditionalCoords = 0;
        if (result.contains("turnsDescription") && result["turnsDescription"].is_array()) {
            for (const auto& turn : result["turnsDescription"]) {
                if (turn.contains("additionalCoordinates") && !turn["additionalCoordinates"].is_null()) {
                    turnsWithAdditionalCoords++;
                }
            }
        }
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << coilString << std::endl;
        std::cerr << repetitions << std::endl;
        std::cerr << proportionPerWindingString << std::endl;
        std::cerr << patternString << std::endl;
        std::cerr << marginPairsString << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::string wind(std::string coilString, size_t repetitions, std::string proportionPerWindingString, std::string patternString, std::string marginPairsString) {
    return wind_impl(coilString, "", repetitions, proportionPerWindingString, patternString, marginPairsString);
}

// The engine's automatic per-winding proportions (physical turn area per
// winding from turns × parallels × wire area) — what wind() uses when no
// proportions are given. Backs the studio's Auto-fit button.
std::string calculate_proportion_per_winding_based_on_wires(std::string coilString) {
    try {
        auto coilJson = json::parse(coilString);
        OpenMagnetics::Coil coil(coilJson, false);
        json result = coil.get_proportion_per_winding_based_on_wires();
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Multi-column variant of wind(): identical, plus the core columns JSON
// (core.processedDescription.columns) so placements into non-main winding
// windows (windingWindow on winding/group/section) can be wound, an optional
// map of hand-drawn section rectangles ({name: {coordinates, dimensions}},
// "" for none) re-imposed after compaction, and an explicit delimit/compact
// switch.
std::string wind_with_columns(std::string coilString, std::string coreColumnsString, size_t repetitions, std::string proportionPerWindingString, std::string patternString, std::string marginPairsString, std::string customSectionRectsString, bool delimitAndCompact) {
    return wind_impl(coilString, coreColumnsString, repetitions, proportionPerWindingString, patternString, marginPairsString, customSectionRectsString, delimitAndCompact);
}

std::string wind_planar(std::string coilString, std::string stackUpString, double borderToWireDistance, std::string wireToWireDistanceString, std::string insulationThicknessString, double coreToLayerDistance) {
    try {
        OpenMagnetics::Settings::GetInstance().set_coil_wind_even_if_not_fit(true);
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(true);
        auto coilJson = json::parse(coilString);
        auto coil = OpenMagnetics::Coil(coilJson, false);
        std::vector<size_t> stackUp = json::parse(stackUpString);
        std::map<std::pair<size_t, size_t>, double> insulationThickness = json::parse(insulationThicknessString).get<std::map<std::pair<size_t, size_t>, double>>();
        std::map<size_t, double> wireToWireDistance = json::parse(wireToWireDistanceString).get<std::map<size_t, double>>();

        coil.set_strict(false);
        coil.wind_planar(stackUp, borderToWireDistance, wireToWireDistance, insulationThickness, coreToLayerDistance);

        if (!coil.get_turns_description()) {
            {
                // ABT #930: name why the wind produced nothing — a bare "Turns not created"
                // reads as a broken winder even when MKF is right to refuse the geometry.
                const auto& fitReason = coil.get_last_fit_failure();
                throw std::runtime_error(fitReason.empty()
                    ? std::string("Turns not created: the winding does not fit its window, and the "
                                  "reason could not be narrowed further.")
                    : "Turns not created. " + fitReason);
            }
        }

        // Explicitly call delimit_and_compact to ensure proper compacting
        coil.delimit_and_compact();

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << coilString << std::endl;
        std::cerr << stackUpString << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::string wind_by_sections(std::string coilString, size_t repetitions, std::string proportionPerWindingString, std::string patternString) {
    try {
        auto coilJson = json::parse(coilString);

        std::vector<double> proportionPerWinding = json::parse(proportionPerWindingString);
        std::vector<size_t> pattern = json::parse(patternString);
        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        OpenMagnetics::Coil coil;

        process_coil_configuration(coil, coilString, repetitions, proportionPerWinding, pattern);

        coil.set_bobbin_from_json(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        if (proportionPerWinding.size() == winding.size()) {
            if (pattern.size() > 0 && repetitions > 0) {
                coil.wind_by_sections(proportionPerWinding, pattern, repetitions);
            }
            else if (repetitions > 0) {
                coil.wind_by_sections(repetitions);
            }
            else {
                coil.wind_by_sections();
            }
        }
        else {
            if (pattern.size() > 0 && repetitions > 0) {
                coil.wind_by_sections(pattern, repetitions);
            }
            else if (repetitions > 0) {
                coil.wind_by_sections(repetitions);
            }
            else {
                coil.wind_by_sections();
            }
        }

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string wind_by_layers(std::string coilString) {
    try {
        auto coilJson = json::parse(coilString);

        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        auto coilSectionsDescription = std::vector<Section>(coilJson["sectionsDescription"]);
        OpenMagnetics::Coil coil;

        process_coil_configuration(coil, coilString);

        coil.set_bobbin_from_json(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.set_sections_description(coilSectionsDescription);
        coil.wind_by_layers();

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string wind_by_turns(std::string coilString) {
    try {
        auto coilJson = json::parse(coilString);

        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        auto coilSectionsDescription = std::vector<Section>(coilJson["sectionsDescription"]);
        auto coilLayersDescription = std::vector<Layer>(coilJson["layersDescription"]);
        OpenMagnetics::Coil coil;

        process_coil_configuration(coil, coilString);

        coil.set_bobbin_from_json(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.set_sections_description(coilSectionsDescription);
        coil.set_layers_description(coilLayersDescription);
        coil.wind_by_turns();

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

static std::string delimit_and_compact_impl(const std::string& coilString, const std::string& coreColumnsString) {
    try {
        auto coilJson = json::parse(coilString);

        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        auto coilSectionsDescription = std::vector<Section>(coilJson["sectionsDescription"]);
        auto coilLayersDescription = std::vector<Layer>(coilJson["layersDescription"]);
        auto coilTurnsDescription = std::vector<Turn>(coilJson["turnsDescription"]);
        OpenMagnetics::Coil coil;

        process_coil_configuration(coil, coilString);

        coil.set_bobbin_from_json(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.set_sections_description(coilSectionsDescription);
        coil.set_layers_description(coilLayersDescription);
        coil.set_turns_description(coilTurnsDescription);
        // The serialized descriptions are at their FINAL multi-window positions;
        // without this, delimit_and_compact re-compacts mirrored-window sections
        // as if they were still in the +x winding frame.
        coil.set_group_window_sides_applied(true);
        if (!coreColumnsString.empty()) {
            // Multi-column placement: sections in non-main winding windows need the
            // core columns to rebuild their lateral wound-column frames.
            std::vector<ColumnElement> coreColumns = json::parse(coreColumnsString);
            coil.set_core_columns(coreColumns);
        }

        // Preserve groupsDescription if it exists
        if (coilJson.contains("groupsDescription") && !coilJson["groupsDescription"].is_null()) {
            auto groupsDescription = std::vector<Group>(coilJson["groupsDescription"]);
            coil.set_groups_description(groupsDescription);
        }

        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(true);

        coil.delimit_and_compact();

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Custom-rectangle re-flow (winding studio): re-run layers+turns INSIDE the
// caller-provided section rectangles — sections are NOT recomputed and the
// compaction pass that would undo the custom placement is NOT run.
std::string wind_layers_and_turns_with_columns(std::string coilString, std::string coreColumnsString) {
    try {
        auto coilJson = json::parse(coilString);

        OpenMagnetics::Settings::GetInstance().set_coil_wind_even_if_not_fit(true);
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(true);

        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        auto coilSectionsDescription = std::vector<Section>(coilJson["sectionsDescription"]);
        OpenMagnetics::Coil coil;

        process_coil_configuration(coil, coilString);

        coil.set_bobbin_from_json(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.set_sections_description(coilSectionsDescription);
        if (coilJson.contains("groupsDescription") && !coilJson["groupsDescription"].is_null()) {
            coil.set_groups_description(std::vector<Group>(coilJson["groupsDescription"]));
        }
        // The provided rectangles are at their FINAL multi-window positions.
        coil.set_group_window_sides_applied(true);
        if (!coreColumnsString.empty()) {
            std::vector<ColumnElement> coreColumns = json::parse(coreColumnsString);
            coil.set_core_columns(coreColumns);
        }

        if (!coil.rewind_layers_and_turns()) {
            {
                // ABT #930: name why the wind produced nothing — a bare "Turns not created"
                // reads as a broken winder even when MKF is right to refuse the geometry.
                const auto& fitReason = coil.get_last_fit_failure();
                throw std::runtime_error(fitReason.empty()
                    ? std::string("Turns not created: the winding does not fit its window, and the "
                                  "reason could not be narrowed further.")
                    : "Turns not created. " + fitReason);
            }
        }

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string delimit_and_compact(std::string coilString) {
    return delimit_and_compact_impl(coilString, "");
}

// Multi-column variant of delimit_and_compact(): identical, plus the core
// columns JSON (core.processedDescription.columns).
std::string delimit_and_compact_with_columns(std::string coilString, std::string coreColumnsString) {
    return delimit_and_compact_impl(coilString, coreColumnsString);
}

std::string get_layers_by_winding_index(std::string coilString, int windingIndex){
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);

        json result = json::array();
        for (auto& layer : coil.get_layers_by_winding_index(windingIndex)) {
            json aux;
            to_json(aux, layer);
            result.push_back(aux);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_layers_by_section(std::string coilString, std::string sectionName){
    try {
        json result = json::array();
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        for (auto& layer : coil.get_layers_by_section(sectionName)) {
            json aux;
            to_json(aux, layer);
            result.push_back(aux);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_sections_description_conduction(std::string coilString){
    try {
        json result = json::array();
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        for (auto& section : coil.get_sections_description_conduction()) {
            json aux;
            to_json(aux, section);
            result.push_back(aux);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

bool are_sections_and_layers_fitting(std::string coilString) {
    try {
        json result = json::array();
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        return coil.are_sections_and_layers_fitting();
    }
    catch (const std::exception &exc) {
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
        return false;
    }
}

std::string add_margin_to_section_by_index(std::string coilString, int sectionIndex, double top_or_left_margin, double bottom_or_right_margin) {
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        coil.add_margin_to_section_by_index(sectionIndex, {top_or_left_margin, bottom_or_right_margin});

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate(std::string inputsString,
                     std::string magneticString,
                     std::string modelsData){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OpenMagnetics::Inputs inputs(json::parse(inputsString));

        auto defaults = OpenMagnetics::Defaults();

        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();

        for (const auto& [key, value] : models) {
        }

        auto reluctanceModelName = OpenMagnetics::defaults.reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }
        auto coreLossesModelName = OpenMagnetics::defaults.coreLossesModelDefault;
        if (models.find("coreLosses") != models.end()) {
            OpenMagnetics::from_json(models["coreLosses"], coreLossesModelName);
        }
        auto coreTemperatureModelName = OpenMagnetics::defaults.coreTemperatureModelDefault;
        if (models.find("coreTemperature") != models.end()) {
            OpenMagnetics::from_json(models["coreTemperature"], coreTemperatureModelName);
        }


        OpenMagnetics::MagneticSimulator magneticSimulator;

        magneticSimulator.set_core_losses_model_name(coreLossesModelName);
        // The core-temperature-model knob moved from MagneticSimulator to Settings
        // (MKF commit 10aa82c: simulator now derives core-loss temperature via the
        // thermal-network model). Set it on Settings to honor the request payload.
        OpenMagnetics::Settings::GetInstance().set_core_temperature_model(coreTemperatureModelName);
        magneticSimulator.set_reluctance_model_name(reluctanceModelName);
        auto mas = magneticSimulator.simulate(inputs, magnetic);

        if (mas.get_outputs()[0].get_winding_losses()) {
        }

        json result;
        to_json(result, mas);

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


bool check_if_fits(std::string bobbinString, double dimension, bool isHorizontalOrRadial) {
    try {
        OpenMagnetics::Bobbin bobbin(json::parse(bobbinString));
        return bobbin.check_if_fits(dimension, isHorizontalOrRadial);
    }
    catch (const std::exception &exc) {
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
        return false;
    }
}

std::string export_magnetic_as_subcircuit(std::string magneticString, double temperature, std::string simulatorString, std::string jsimba) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        OpenMagnetics::CircuitSimulatorExporterModels simulator;
        from_json(simulatorString, simulator);

        switch(simulator) {
            case OpenMagnetics::CircuitSimulatorExporterModels::SIMBA:
                {
                    std::string subcircuit;
                    if (jsimba != "") {
                        subcircuit = OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_subcircuit(magnetic, OpenMagnetics::Defaults().measurementFrequency, temperature, std::nullopt, jsimba);
                    }
                    else {
                        subcircuit = OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_subcircuit(magnetic, OpenMagnetics::Defaults().measurementFrequency, temperature);
                    }
                    return subcircuit;
                    break;
                }
            case OpenMagnetics::CircuitSimulatorExporterModels::LTSPICE:
                return OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_subcircuit(magnetic, OpenMagnetics::Defaults().measurementFrequency, temperature);
                break;
            case OpenMagnetics::CircuitSimulatorExporterModels::NGSPICE:
                return OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_subcircuit(magnetic, OpenMagnetics::Defaults().measurementFrequency, temperature);
                break;
            case OpenMagnetics::CircuitSimulatorExporterModels::NL5:
                return OpenMagnetics::CircuitSimulatorExporterNl5Model().export_magnetic_as_subcircuit(magnetic, OpenMagnetics::Defaults().measurementFrequency, temperature);
            case OpenMagnetics::CircuitSimulatorExporterModels::PLECS:
                return OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_subcircuit(magnetic, OpenMagnetics::Defaults().measurementFrequency, temperature);
        }


    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


std::string export_magnetic_as_symbol(std::string magneticString, std::string simulatorString, std::string jsimba) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        OpenMagnetics::CircuitSimulatorExporterModels simulator;
        from_json(simulatorString, simulator);

        switch(simulator) {
            case OpenMagnetics::CircuitSimulatorExporterModels::SIMBA:
                break;
            case OpenMagnetics::CircuitSimulatorExporterModels::LTSPICE:
                return OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_symbol(magnetic);
                break;
            case OpenMagnetics::CircuitSimulatorExporterModels::NGSPICE:
                break;
            case OpenMagnetics::CircuitSimulatorExporterModels::NL5:
                break;
            case OpenMagnetics::CircuitSimulatorExporterModels::PLECS:
                return OpenMagnetics::CircuitSimulatorExporter(simulator).export_magnetic_as_symbol(magnetic);
                break;
        }

        return "";

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


void calculate_ac_resistance_coefficients_per_winding(std::string magneticString, double temperature) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        auto coefficientsPerWinding = OpenMagnetics::CircuitSimulatorExporter(OpenMagnetics::CircuitSimulatorExporterModels::LTSPICE).calculate_ac_resistance_coefficients_per_winding(magnetic, temperature);
        for (auto coefficients : coefficientsPerWinding) {
            for (auto coefficient : coefficients) {
                std::cout << "coefficient: " << coefficient << std::endl;
            }
        }

    }
    catch (const std::exception &exc) {
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
    }
}


std::string sweep_impedance_over_frequency(std::string magneticString, double start, double stop, size_t numberElements, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        auto impedanceOverFrequency = OpenMagnetics::Sweeper::sweep_impedance_over_frequency(magnetic, start, stop, numberElements, mode, title);

        json result;
        to_json(result, impedanceOverFrequency);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


std::string sweep_q_factor_over_frequency(std::string magneticString, double start, double stop, size_t numberElements, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        auto impedanceOverFrequency = OpenMagnetics::Sweeper::sweep_q_factor_over_frequency(magnetic, start, stop, numberElements, mode, title);

        json result;
        to_json(result, impedanceOverFrequency);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


std::string sweep_winding_resistance_over_frequency(std::string magneticString, double start, double stop, size_t numberElements, size_t windingIndex, double temperature, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        auto resistanceOverFrequency = OpenMagnetics::Sweeper::sweep_winding_resistance_over_frequency(magnetic, start, stop, numberElements, windingIndex, temperature, mode, title);

        json result;
        to_json(result, resistanceOverFrequency);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string sweep_resistance_over_frequency(std::string magneticString, double start, double stop, size_t numberElements, double temperature, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        auto resistanceOverFrequency = OpenMagnetics::Sweeper::sweep_resistance_over_frequency(magnetic, start, stop, numberElements, temperature, mode, title);

        json result;
        to_json(result, resistanceOverFrequency);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string sweep_magnetizing_inductance_over_frequency(std::string magneticString, double start, double stop, size_t numberElements, double temperature, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        auto magnetizingInductanceOverFrequency = OpenMagnetics::Sweeper::sweep_magnetizing_inductance_over_frequency(magnetic, start, stop, numberElements, temperature, mode, title);

        json result;
        to_json(result, magnetizingInductanceOverFrequency);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string sweep_magnetizing_inductance_over_temperature(std::string magneticString, double start, double stop, size_t numberElements, double frequency, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        auto magnetizingInductanceOverTemperature = OpenMagnetics::Sweeper::sweep_magnetizing_inductance_over_temperature(magnetic, start, stop, numberElements, frequency, mode, title);

        json result;
        to_json(result, magnetizingInductanceOverTemperature);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string sweep_magnetizing_inductance_over_dc_bias(std::string magneticString, double start, double stop, size_t numberElements, double temperature, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        auto magnetizingInductanceOverDcBias = OpenMagnetics::Sweeper::sweep_magnetizing_inductance_over_dc_bias(magnetic, start, stop, numberElements, temperature, mode, title);

        json result;
        to_json(result, magnetizingInductanceOverDcBias);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string sweep_core_losses_over_frequency(std::string magneticString, std::string operatingPointString, double start, double stop, size_t numberElements, double temperature, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        auto resistanceOverFrequency = OpenMagnetics::Sweeper::sweep_core_losses_over_frequency(magnetic, operatingPoint, start, stop, numberElements, temperature, mode, title);

        json result;
        to_json(result, resistanceOverFrequency);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string sweep_winding_losses_over_frequency(std::string magneticString, std::string operatingPointString, double start, double stop, size_t numberElements, double temperature, std::string mode, std::string title) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        auto sweep = OpenMagnetics::Sweeper::sweep_winding_losses_over_frequency(magnetic, operatingPoint, start, stop, numberElements, temperature, mode, title);

        json result;
        to_json(result, sweep);

        return result.dump(4);

    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// ABT #166: material-level volumetric-loss sampler. Lets the frontend plot
// Pv(f) for ANY loss model (Steinmetz, iGSE, Roshen, loss maps, proprietary)
// from engine numbers, without re-implementing magnetics math in JS.
// materialString: material name in the DB or a full CoreMaterial JSON.
// methodString: one of the CoreLossesModels json names ("Steinmetz", "IGSE",
// "Roshen", "Proprietary", ...) or "" to use the material's best-ranked method.
std::string sweep_volumetric_losses_over_frequency(std::string materialString, double temperature, double magneticFluxDensityPeak, double start, double stop, size_t numberElements, std::string methodString) {
    try {
        CoreMaterial material = materialString.size() > 0 && materialString.front() == '{'
            ? CoreMaterial(json::parse(materialString))
            : OpenMagnetics::find_core_material_by_name(materialString);

        std::shared_ptr<OpenMagnetics::CoreLossesModel> model;
        if (methodString == "") {
            auto methods = OpenMagnetics::CoreLossesModel::get_methods(material);
            if (methods.size() == 0) {
                throw std::runtime_error("No core losses method available for material " + material.get_name());
            }
            model = OpenMagnetics::CoreLossesModel::factory(methods[0]);
        }
        else {
            OpenMagnetics::CoreLossesModels method;
            OpenMagnetics::from_json(json(methodString), method);
            model = OpenMagnetics::CoreLossesModel::factory(method);
        }

        json points = json::array();
        for (size_t index = 0; index < numberElements; ++index) {
            double frequency = numberElements > 1
                ? start * pow(stop / start, static_cast<double>(index) / (numberElements - 1))
                : start;
            auto waveform = OpenMagnetics::Inputs::create_waveform(WaveformLabel::SINUSOIDAL, 2 * magneticFluxDensityPeak, frequency);
            SignalDescriptor magneticFluxDensity;
            magneticFluxDensity.set_waveform(waveform);
            magneticFluxDensity.set_processed(OpenMagnetics::Inputs::calculate_processed_data(waveform, frequency));
            OperatingPointExcitation excitation;
            excitation.set_frequency(frequency);
            excitation.set_magnetic_flux_density(magneticFluxDensity);

            json point;
            point["frequency"] = frequency;
            point["value"] = model->get_core_volumetric_losses(material, excitation, temperature);
            points.push_back(point);
        }

        json result;
        result["methodUsed"] = model->get_model_name();
        result["temperature"] = temperature;
        result["magneticFluxDensityPeak"] = magneticFluxDensityPeak;
        result["points"] = points;
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

size_t load_core_materials(std::string fileToLoad){
    try {
        if (fileToLoad != "") {
            OpenMagnetics::load_core_materials(fileToLoad);
        }
        else {
            OpenMagnetics::load_core_materials();
        }

        return OpenMagnetics::coreMaterialDatabase.size();
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return -1;
    }
}

size_t load_core_shapes(std::string fileToLoad){
    try {
        if (fileToLoad != "") {
            OpenMagnetics::load_core_shapes(true, fileToLoad);
        }
        else {
            OpenMagnetics::load_core_shapes();
        }
        return OpenMagnetics::coreShapeDatabase.size();
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return -1;
    }
}

size_t load_wires(std::string fileToLoad){
    try {
        if (fileToLoad != "") {
            OpenMagnetics::load_wires(fileToLoad);
        }
        else {
            OpenMagnetics::load_wires();
        }
        return OpenMagnetics::wireDatabase.size();
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return -1;
    }
}

size_t load_cores(std::string fileToLoad, bool includeToroids, bool useOnlyCoresInStock){
    try {
        if (fileToLoad != "") {
            OpenMagnetics::load_cores(fileToLoad);
        }
        else {
            OpenMagnetics::Settings::GetInstance().set_use_toroidal_cores(includeToroids);
            OpenMagnetics::Settings::GetInstance().set_use_only_cores_in_stock(useOnlyCoresInStock);
            OpenMagnetics::load_cores();
        }
        return OpenMagnetics::coreDatabase.size();
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return -1;
    }
}

void clear_loaded_cores(){
    OpenMagnetics::clear_loaded_cores();
}

void clear_databases(){
    OpenMagnetics::clear_databases();
}

bool is_core_material_database_empty(){
    return OpenMagnetics::coreMaterialDatabase.size() == 0;
}

bool is_core_shape_database_empty(){
    return OpenMagnetics::coreShapeDatabase.size() == 0;
}

bool is_wire_database_empty(){
    return OpenMagnetics::wireDatabase.size() == 0;
}

std::vector<double> get_maximum_dimensions(std::string magneticString){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        return magnetic.get_maximum_dimensions();
    }
    catch (const std::exception &exc) {
        std::cerr << "Exception: " + std::string{exc.what()} << std::endl;
        return {};
    }
}

// Shared implementation behind calculate_advised_cores and its _with_context
// twin: identical parsing, suppression handling, scoring and output shape;
// ctx=nullptr means "public catalogs" (the classic behavior).
std::string calculate_advised_cores_impl(std::string inputsString, std::string weightsString, int maximumNumberResults, std::string coreModeString,
                                         const OpenMagnetics::LibraryContext* ctx, const OpenMagnetics::AdviserConstraints& adviserConstraints){
    try {
        // Drain stale entries (and enable log capture on first use) so the
        // log returned with this run's results covers exactly this run.
        OpenMagnetics::read_log();
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);

        OpenMagnetics::Inputs inputs(json::parse(inputsString));
        OpenMagnetics::CoreAdviser::CoreAdviserModes coreMode;
        from_json(coreModeString, coreMode);
        std::map<std::string, double> weightsKeysString = json::parse(weightsString);
        std::map<OpenMagnetics::CoreAdviser::CoreAdviserFilters, double> weights;

        bool filterMode = bool(inputs.get_design_requirements().get_minimum_impedance());

        // Detect CMC/DMC (interference-suppression application). The wizard
        // tags DesignRequirements; this standalone entry point used to ignore
        // that tag and leave CoreAdviser in POWER mode, silently routing
        // through the wrong filter flow. Mirror the MagneticAdviser policy
        // here: if INTERFERENCE_SUPPRESSION is set, switch to AVAILABLE_CORES
        // (toroidal EMI parts come off a catalog, no gap-grinding) and force
        // toroidal-only settings.
        const bool isSuppression =
            inputs.get_design_requirements().get_application().has_value()
            && inputs.get_design_requirements().get_application().value()
               == "interferenceSuppression";

        if (filterMode || isSuppression) {
            OpenMagnetics::Settings::GetInstance().set_use_toroidal_cores(true);
            OpenMagnetics::Settings::GetInstance().set_use_only_cores_in_stock(false);
            OpenMagnetics::Settings::GetInstance().set_use_concentric_cores(false);
        }
        if (isSuppression && coreMode == OpenMagnetics::CoreAdviser::CoreAdviserModes::STANDARD_CORES) {
            coreMode = OpenMagnetics::CoreAdviser::CoreAdviserModes::AVAILABLE_CORES;
        }

        double externalSum = 0;
        for (auto const& pair : weightsKeysString) {
            externalSum += pair.second;
        }

        std::cout << "Parsed weights:" << std::endl;
        for (auto const& [filterName, weight] : weightsKeysString) {
            OpenMagnetics::CoreAdviser::CoreAdviserFilters filter;
            OpenMagnetics::from_json(filterName, filter);
            weights[filter] = weight / externalSum;
            std::cout << "  " << filterName << " -> " << (weight / externalSum) << std::endl;
        }

        OpenMagnetics::CoreAdviser coreAdviser;
        coreAdviser.set_mode(coreMode);
        // Critical: without this, the CoreAdviser defaults to POWER and
        // picks the wrong filter-flow branch for CMC/DMC (no impedance
        // filter, no interference-suppression material pruning).
        if (isSuppression) {
            coreAdviser.set_application(MAS::MagneticApplication::INTERFERENCE_SUPPRESSION);
        }
        std::cout << "[DEBUG] CoreAdviser mode set to: " << (int)coreMode << " (0=AVAILABLE, 1=STANDARD, 2=CUSTOM)" << std::endl;
        std::cout << "[DEBUG] Requesting " << maximumNumberResults << " results" << std::endl;
        std::cout << "[DEBUG] coreShapeDatabase.size() = " << OpenMagnetics::coreShapeDatabase.size() << std::endl;
        std::cout << "[DEBUG] coreMaterialDatabase.size() = " << OpenMagnetics::coreMaterialDatabase.size() << std::endl;
        std::cout << "[DEBUG] useToroidalCores = " << OpenMagnetics::Settings::GetInstance().get_use_toroidal_cores() << std::endl;
        std::cout << "[DEBUG] useConcentricCores = " << OpenMagnetics::Settings::GetInstance().get_use_concentric_cores() << std::endl;
        // ctx==nullptr takes the ORIGINAL base-overload path: the ctx overload
        // is documented as equivalent for nullptr+empty constraints but is not
        // (returns zero results) — MKF bug, ABT-tracked. Never route existing
        // users through it.
        std::vector<std::pair<OpenMagnetics::Mas, double>> masMagnetics;
        if (ctx == nullptr) {
            masMagnetics = coreAdviser.get_advised_core(inputs, weights, maximumNumberResults);
        } else {
            masMagnetics = coreAdviser.get_advised_core(inputs, weights, maximumNumberResults, ctx, adviserConstraints);
        }
        auto log = OpenMagnetics::read_log();
        std::cout << "[DEBUG] MKF Log:\n" << log << std::endl;
        std::cout << "[DEBUG] Results count: " << masMagnetics.size() << std::endl;
        for (size_t i = 0; i < masMagnetics.size() && i < 10; ++i) {
            auto& magnetic = masMagnetics[i].first.get_magnetic();
            std::string coreName = magnetic.get_core().get_name() ? magnetic.get_core().get_name().value() : "unnamed";
            std::cout << "[DEBUG] Result " << i << ": " << coreName << " - Score: " << masMagnetics[i].second << std::endl;
        }
        auto scoring = coreAdviser.get_scorings();
        std::map<std::string, std::map<std::string, double>> filteredScoring;

        json results = json();
        results["data"] = json::array();
        for (auto& masMagnetic : masMagnetics) {
            std::string name = masMagnetic.first.get_magnetic().get_manufacturer_info().value().get_reference().value();
            auto mas = masMagnetic.first;
            // Extra outputs
            {
                OpenMagnetics::MagnetizingInductance magnetizingInductanceModel;
                for (size_t operatingPointIndex = 0; operatingPointIndex < inputs.get_operating_points().size(); ++operatingPointIndex) {
                    auto operatingPoint = inputs.get_operating_point(operatingPointIndex);
                    auto magnetizingInductanceOutput = magnetizingInductanceModel.calculate_inductance_from_number_turns_and_gapping(mas.get_magnetic().get_core(), mas.get_magnetic().get_coil(), &operatingPoint);
                    if (mas.get_mutable_outputs()[operatingPointIndex].get_inductance()) {
                        auto magnetizingInductanceOutputEnergy = mas.get_mutable_outputs()[operatingPointIndex].get_inductance()->get_magnetizing_inductance();
                        magnetizingInductanceOutput.set_maximum_magnetic_energy_core(magnetizingInductanceOutputEnergy.get_maximum_magnetic_energy_core());
                        InductanceOutput inductanceOutput = *mas.get_mutable_outputs()[operatingPointIndex].get_inductance();
                        inductanceOutput.set_magnetizing_inductance(magnetizingInductanceOutput);
                        mas.get_mutable_outputs()[operatingPointIndex].set_inductance(inductanceOutput);
                    }
                    masMagnetic.first = mas;
                }
            }

            json result;
            json masJson;
            to_json(masJson, masMagnetic.first);
            result["mas"] = masJson;
            // result["weightedTotalScoring"] = masMagnetic.second;
            result["weightedTotalScoring"] = scoring[name][OpenMagnetics::CoreAdviser::CoreAdviserFilters::COST] + scoring[name][OpenMagnetics::CoreAdviser::CoreAdviserFilters::EFFICIENCY] + scoring[name][OpenMagnetics::CoreAdviser::CoreAdviserFilters::DIMENSIONS];
            result["scoringPerFilter"] = json();

            for (size_t index = 0; index < magic_enum::enum_count<OpenMagnetics::CoreAdviser::CoreAdviserFilters>(); ++index) {
                auto filter = static_cast<OpenMagnetics::CoreAdviser::CoreAdviserFilters>(index);
                auto filterString = OpenMagnetics::to_string(filter);
                result["scoringPerFilter"][filterString] = scoring[name][filter];
            };
            results["data"].push_back(result);
        }
        results["log"] = log;

        return results.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << inputsString << std::endl;
        std::cerr << weightsString << std::endl;
        std::cerr << maximumNumberResults << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advised_cores(std::string inputsString, std::string weightsString, int maximumNumberResults, std::string coreModeString){
    return calculate_advised_cores_impl(inputsString, weightsString, maximumNumberResults, coreModeString,
                                        nullptr, OpenMagnetics::AdviserConstraints{});
}

std::string calculate_advised_sections(std::string masString, std::string patternString, int repetitions){
    try {
        OpenMagnetics::Mas mas(json::parse(masString));
        json patternJson = json::parse(patternString);
        std::vector<size_t> pattern; 
        for (auto& elem : patternJson) {
            pattern.push_back(elem);
        }

        auto bobbin = mas.get_magnetic().get_coil().get_bobbin();
        if (std::holds_alternative<std::string>(bobbin)) {
            auto bobbinString = std::get<std::string>(bobbin);
            if (bobbinString == "Dummy") {
                mas.get_mutable_magnetic().get_mutable_coil().set_bobbin(OpenMagnetics::Bobbin::create_quick_bobbin(mas.get_mutable_magnetic().get_mutable_core()));
            }
        }
        for (size_t windingIndex = 0; windingIndex < mas.get_magnetic().get_coil().get_functional_description().size(); ++windingIndex) {
            mas.get_mutable_magnetic().get_mutable_coil().get_mutable_functional_description()[windingIndex].set_wire("Dummy");
        }

        auto sections = OpenMagnetics::CoilAdviser().get_advised_sections(mas, pattern, repetitions);
        json result = json::array();
        for (auto& section : sections) {
            json aux;
            to_json(aux, section);
            result.push_back(aux);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advised_coil(std::string masString){
    try {
        // Log the raw input string for debugging
        std::cout << "=== INPUT_JSON_START ===" << std::endl;
        std::cout << masString << std::endl;
        std::cout << "=== INPUT_JSON_END ===" << std::endl;
        
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Mas mas(json::parse(masString));

        for (size_t windingIndex = 0; windingIndex < mas.get_magnetic().get_coil().get_functional_description().size(); ++windingIndex) {
            mas.get_mutable_magnetic().get_mutable_coil().get_mutable_functional_description()[windingIndex].set_wire("Dummy");
        }
        mas.get_mutable_magnetic().get_mutable_coil().set_turns_description(std::nullopt);
        mas.get_mutable_magnetic().get_mutable_coil().set_layers_description(std::nullopt);
        mas.get_mutable_magnetic().get_mutable_coil().set_sections_description(std::nullopt);
        mas.get_mutable_magnetic().get_mutable_coil().set_groups_description(std::nullopt);

        OpenMagnetics::CoilAdviser coilAdviser;
        auto masMagneticsWithCoil = coilAdviser.get_advised_coil(mas, 1);

        if (masMagneticsWithCoil.size() > 0) {
            json result = json();
            to_json(result, masMagneticsWithCoil[0]);
        std::cout << "=== OUTPUT_JSON_START ===" << std::endl;
        std::cout << result << std::endl;
        std::cout << "=== OUTPUT_JSON_END ===" << std::endl;
            return result.dump(4);
        }
        else{
            std::cerr << masString << std::endl;
            return "Exception: No coil found";
        }
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advised_wires(std::string windingString,
                                    std::string sectionString,
                                    std::string currentString,
                                    std::string solidInsulationRequirementsString,
                                    double temperature,
                                    uint8_t numberSections,
                                    size_t maximumNumberResults,
                                    bool usePlanarWires){
    try {
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Winding winding(json::parse(windingString));
        OpenMagnetics::WireSolidInsulationRequirements wireSolidInsulationRequirements(json::parse(solidInsulationRequirementsString));
        Section section(json::parse(sectionString));
        SignalDescriptor current(json::parse(currentString));

        OpenMagnetics::WireAdviser wireAdviser;
        wireAdviser.set_wire_solid_insulation_requirements(wireSolidInsulationRequirements);
        std::vector<std::pair<OpenMagnetics::Winding, double>> windingsWithScoring;
        
        if (usePlanarWires) {
            windingsWithScoring = wireAdviser.get_advised_planar_wire(winding, section, current, temperature, numberSections, maximumNumberResults);
        }
        else {
            windingsWithScoring = wireAdviser.get_advised_wire(winding, section, current, temperature, numberSections, maximumNumberResults);
        }

        json results = json();
        results["data"] = json::array();
        for (auto& [winding, scoring] : windingsWithScoring) {
            json result;
            json windingJson;
            to_json(windingJson, winding);
            result["winding"] = windingJson;
            result["scoring"] = scoring;
            results["data"].push_back(result);
        }

        return results.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_solid_insulation_requirements_for_wires(std::string inputsString, std::string patternString, int repetitions) {
    try {
        OpenMagnetics::Inputs inputs(json::parse(inputsString));
        json patternJson = json::parse(patternString);
        std::vector<size_t> pattern; 
        for (auto& elem : patternJson) {
            pattern.push_back(elem);
        }

        auto results = json::array();
        auto solidInsulationRequirementsCombinations = OpenMagnetics::InsulationCoordinator().get_solid_insulation_requirements_for_wires(inputs, pattern, repetitions);
        for (auto solidInsulationRequirementsCombination : solidInsulationRequirementsCombinations) {
            auto aux = json::array();
            for (auto solidInsulationRequirementsForWires : solidInsulationRequirementsCombination) {
                json solidInsulationRequirementsForWiresJson;
                to_json(solidInsulationRequirementsForWiresJson, solidInsulationRequirementsForWires);
                aux.push_back(solidInsulationRequirementsForWiresJson);
            }
            results.push_back(aux);
        }
        return results.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Shared implementation behind calculate_advised_magnetics and its
// _with_context twin (same parsing/suppression flow/output shape; ctx=nullptr
// = public catalogs).
std::string calculate_advised_magnetics_impl(std::string inputsString, std::string weightsString, int maximumNumberResults, std::string coreModeString,
                                             const OpenMagnetics::LibraryContext* ctx, const OpenMagnetics::AdviserConstraints& adviserConstraints){
    try {
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Inputs inputs(json::parse(inputsString));

        OpenMagnetics::CoreAdviser::CoreAdviserModes coreMode;
        from_json(coreModeString, coreMode);

        std::map<std::string, double> weightsKeysString = json::parse(weightsString);
        std::map<OpenMagnetics::MagneticFilters, double> weights;

        double externalSum = 0;
        for (auto const& pair : weightsKeysString) {
            externalSum += pair.second;
        }

        for (auto const& [filterName, weight] : weightsKeysString) {
            OpenMagnetics::MagneticFilters filter;
            OpenMagnetics::from_json(filterName, filter);
            weights[filter] = weight / externalSum;
        }

        OpenMagnetics::MagneticAdviser magneticAdviser;
        magneticAdviser.set_core_mode(coreMode);

        // For interference-suppression (CMC/DMC) the user's COST/LOSSES/
        // DIMENSIONS-only weight set is insufficient — the default weights-
        // to-filter expansion in MagneticAdviser.cpp:152 doesn't include
        // CORE_MINIMUM_IMPEDANCE, so tiny toroids that can't meet the |Z|
        // spec slip through. Build the filter flow explicitly and force
        // impedance + leakage-inductance filters in.
        bool hasApp = inputs.get_design_requirements().get_application().has_value();
        const bool isSuppressionFlow = hasApp
            && inputs.get_design_requirements().get_application().value()
               == "interferenceSuppression";
        std::vector<std::pair<OpenMagnetics::Mas, double>> masMagnetics;
        if (isSuppressionFlow) {
            double wCost = weights.count(OpenMagnetics::MagneticFilters::COST)
                ? weights[OpenMagnetics::MagneticFilters::COST] : 1.0;
            double wLosses = weights.count(OpenMagnetics::MagneticFilters::LOSSES)
                ? weights[OpenMagnetics::MagneticFilters::LOSSES] : 1.0;
            double wDims = weights.count(OpenMagnetics::MagneticFilters::DIMENSIONS)
                ? weights[OpenMagnetics::MagneticFilters::DIMENSIONS] : 1.0;
            // CORE_MINIMUM_IMPEDANCE is the make-or-break filter for EMI
            // suppression: without it the adviser happily picks cores that
            // don't meet |Z| spec. Give it the highest weight.
            // LEAKAGE_INDUCTANCE rewards tight CM coupling on the chosen
            // magnetic (k → 1) so we get a proper CMC.
            std::vector<OpenMagnetics::MagneticFilterOperation> cmcFilterFlow{
                OpenMagnetics::MagneticFilterOperation(OpenMagnetics::MagneticFilters::CORE_MINIMUM_IMPEDANCE, true, true, true, std::max(1.0, wDims * 2.0)),
                OpenMagnetics::MagneticFilterOperation(OpenMagnetics::MagneticFilters::COST,       true, true, wCost),
                OpenMagnetics::MagneticFilterOperation(OpenMagnetics::MagneticFilters::LOSSES,     true, true, wLosses),
                OpenMagnetics::MagneticFilterOperation(OpenMagnetics::MagneticFilters::DIMENSIONS, true, true, wDims),
                OpenMagnetics::MagneticFilterOperation(OpenMagnetics::MagneticFilters::LEAKAGE_INDUCTANCE, true, true, wDims),
                // Manufacturability proxy: penalises low-µ candidates that need an
                // absurd turn count (e.g. powder on a CMC) even when their size/cost
                // are attractive. Score = N_total × max(W, H, D); linear, inverted.
                // Weight is 2× the efficiency weight so the N×dim penalty has enough
                // magnitude to overcome the cost/size advantage powder cores often have.
                OpenMagnetics::MagneticFilterOperation(OpenMagnetics::MagneticFilters::TURN_COUNT, true, false, std::max(wLosses, wDims)),
            };
            if (ctx == nullptr) {
                masMagnetics = magneticAdviser.get_advised_magnetic(inputs, cmcFilterFlow, maximumNumberResults);
            } else {
                masMagnetics = magneticAdviser.get_advised_magnetic(inputs, cmcFilterFlow, maximumNumberResults, ctx, adviserConstraints);
            }
        }
        else {
            if (ctx == nullptr) {
                masMagnetics = magneticAdviser.get_advised_magnetic(inputs, weights, maximumNumberResults);
            } else {
                masMagnetics = magneticAdviser.get_advised_magnetic(inputs, weights, maximumNumberResults, ctx, adviserConstraints);
            }
        }
        // auto log = magneticAdviser.read_log();
        auto scorings = magneticAdviser.get_scorings();

        json results = json();
        results["data"] = json::array();
        for (auto& [masMagnetic, scoring] : masMagnetics) {
            // Safely get the name with optional checks
            std::string name;
            auto& magnetic = masMagnetic.get_magnetic();
            auto manufacturerInfo = magnetic.get_manufacturer_info();
            if (!manufacturerInfo) {
                name = "unnamed";
            } else {
                auto reference = manufacturerInfo.value().get_reference();
                if (!reference) {
                    name = "unnamed";
                } else {
                    name = reference.value();
                }
            }

            json result;
            json masJson;
            to_json(masJson, masMagnetic);
            result["mas"] = masJson;
            result["weightedTotalScoring"] = scorings[name][OpenMagnetics::MagneticFilters::COST] + scorings[name][OpenMagnetics::MagneticFilters::LOSSES] + scorings[name][OpenMagnetics::MagneticFilters::DIMENSIONS];
            result["scoringPerFilter"] = json();
            for (size_t index = 0; index < magic_enum::enum_count<OpenMagnetics::MagneticFilters>(); ++index) {
                auto filter = static_cast<OpenMagnetics::MagneticFilters>(index);
                auto filterString = OpenMagnetics::to_string(filter);
                if (scorings[name].count(filter)) {
                    result["scoringPerFilter"][filterString] = scorings[name][filter];
                }
            };
            results["data"].push_back(result);
        }

        sort(results["data"].begin(), results["data"].end(), [](json& b1, json& b2) {
            return b1["weightedTotalScoring"] > b2["weightedTotalScoring"];
        });

        return results.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advised_magnetics(std::string inputsString, std::string weightsString, int maximumNumberResults, std::string coreModeString){
    return calculate_advised_magnetics_impl(inputsString, weightsString, maximumNumberResults, coreModeString,
                                            nullptr, OpenMagnetics::AdviserConstraints{});
}

std::string calculate_advised_magnetics_from_catalog(std::string inputsString, std::string catalogString, int maximumNumberResults){
    try {
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Inputs inputs(json::parse(inputsString));
        std::map<OpenMagnetics::MagneticFilters, double> weights;

        std::vector <OpenMagnetics::Magnetic> catalog;

        for (auto& catalogSubstring : OpenMagnetics::split(catalogString, "\n")) {
            OpenMagnetics::Magnetic magnetic(json::parse(catalogSubstring));
            catalog.push_back(magnetic);
        }

        OpenMagnetics::MagneticAdviser magneticAdviser;
        auto masMagnetics = magneticAdviser.get_advised_magnetic(inputs, catalog, maximumNumberResults);

        auto scorings = magneticAdviser.get_scorings();

        json results = json();
        results["data"] = json::array();
        for (auto& [masMagnetic, scoring] : masMagnetics) {
            std::string name = masMagnetic.get_magnetic().get_manufacturer_info().value().get_reference().value();
            json result;
            json masJson;
            to_json(masJson, masMagnetic);
            result["mas"] = masJson;
            result["scoring"] = scoring;
            results["data"].push_back(result);
        }

        sort(results["data"].begin(), results["data"].end(), [](json& b1, json& b2) {
            return b1["scoring"] > b2["scoring"];
        });

        return results.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << inputsString << std::endl;
        std::cerr << catalogString << std::endl;
        std::cerr << maximumNumberResults << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advised_magnetics_from_cache(std::string inputsString, std::string filterFlowString, int maximumNumberResults){
    try {
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Inputs inputs(json::parse(inputsString));

        std::vector<OpenMagnetics::MagneticFilterOperation> filterFlow;
        json filterFlowJson = json::parse(filterFlowString);
        for (auto filterJson : filterFlowJson) {
            OpenMagnetics::MagneticFilterOperation filter(filterJson);
            filterFlow.push_back(filter);
        }

        if (OpenMagnetics::magneticsCache.size() == 0) {
            return "Exception: No magnetics found in cache";
        }

        OpenMagnetics::MagneticAdviser magneticAdviser;
        auto masMagnetics = magneticAdviser.get_advised_magnetic(inputs, OpenMagnetics::magneticsCache.get(), filterFlow, maximumNumberResults);

        auto scorings = magneticAdviser.get_scorings();

        json results = json();
        results["data"] = json::array();
        for (auto& [masMagnetic, scoring] : masMagnetics) {
            std::string name = masMagnetic.get_magnetic().get_manufacturer_info().value().get_reference().value();
            json result;
            json masJson;
            to_json(masJson, masMagnetic);
            result["mas"] = masJson;
            result["scoring"] = scoring;
            results["data"].push_back(result);
        }

        sort(results["data"].begin(), results["data"].end(), [](json& b1, json& b2) {
            return b1["scoring"] > b2["scoring"];
        });

        return results.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << inputsString << std::endl;
        std::cerr << filterFlowString << std::endl;
        std::cerr << maximumNumberResults << std::endl;
        return "Exception: " + std::string{exc.what()};
    }
}

std::vector<std::string> get_available_core_filters(){
    std::vector<std::string> filters;
    for (size_t index = 0; index < magic_enum::enum_count<OpenMagnetics::CoreAdviser::CoreAdviserFilters>(); ++index) {
        auto filter = static_cast<OpenMagnetics::CoreAdviser::CoreAdviserFilters>(index);

        filters.push_back(OpenMagnetics::to_string(filter));
    }
    return filters;
}

std::string calculate_leakage_inductance(std::string magneticString, double frequency, size_t sourceIndex){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));

        auto leakageInductanceOutput = OpenMagnetics::LeakageInductance().calculate_leakage_inductance_all_windings(magnetic, frequency, sourceIndex);

        json result;
        to_json(result, leakageInductanceOutput);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_inductance_matrix(std::string magneticString, double frequency, std::string modelsData){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::Inductance inductance(reluctanceModelName);
        auto inductanceMatrix = inductance.calculate_inductance_matrix(magnetic, frequency);

        json result;
        to_json(result, inductanceMatrix);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_coupling_coefficient_matrix(std::string magneticString, double frequency, std::string modelsData){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::Inductance inductance(reluctanceModelName);
        
        auto& functionalDescription = magnetic.get_coil().get_functional_description();
        size_t numWindings = functionalDescription.size();
        
        ScalarMatrixAtFrequency result;
        result.set_frequency(frequency);
        
        std::map<std::string, std::map<std::string, DimensionWithTolerance>> magnitude;
        
        // Calculate coupling coefficient matrix
        for (size_t i = 0; i < numWindings; ++i) {
            std::string windingName_i = functionalDescription[i].get_name();
            
            for (size_t j = 0; j < numWindings; ++j) {
                std::string windingName_j = functionalDescription[j].get_name();
                
                double k = inductance.calculate_coupling_coefficient(magnetic, i, j, frequency);
                
                DimensionWithTolerance dimValue;
                dimValue.set_nominal(k);
                magnitude[windingName_i][windingName_j] = dimValue;
            }
        }
        
        result.set_magnitude(magnitude);

        json jsonResult;
        to_json(jsonResult, result);
        return jsonResult.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_leakage_inductance_matrix(std::string magneticString, double frequency, std::string modelsData){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto reluctanceModelName = OpenMagnetics::Defaults().reluctanceModelDefault;
        // ABT #1085: the web passes the reluctance model as "gapReluctance" (the key
        // the model settings use); "reluctance" is kept for older callers.
        if (models.find("gapReluctance") != models.end()) {
            OpenMagnetics::from_json(models["gapReluctance"], reluctanceModelName);
        }
        else if (models.find("reluctance") != models.end()) {
            OpenMagnetics::from_json(models["reluctance"], reluctanceModelName);
        }

        OpenMagnetics::Inductance inductance(reluctanceModelName);
        auto leakageInductanceMatrix = inductance.calculate_leakage_inductance_matrix(magnetic, frequency);

        json result;
        to_json(result, leakageInductanceMatrix);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// The capacitance entry points below used to take a bare COIL, so the turn-to-core
// network (ABT #848: the exact cylinder-over-plane element, the core image factor from
// the material's permittivity, and the real per-turn wire-to-core gap) could never act —
// there was no core to act on. The panel was therefore structurally unable to agree with
// the impedance sweep, which builds a full Magnetic. Measured on production before this:
// designs whose resonance implied 43 nF and 220 nF of stray capacitance, where a wound
// component is pF-class.
//
// Accepting a MAGNETIC costs nothing at the call site and keeps the arity identical, so
// no caller can be handed the wrong number of arguments (the failure mode that took the
// 3D view down twice this month). A bare coil is still accepted: if the JSON has a "coil"
// key it is a Magnetic, otherwise it is the coil itself.
//
// frequency is deliberately NOT threaded here. It only sets the core image factor beta,
// which is 1 for MnZn, nanocrystalline and any conductor — the impedance path passes its
// own resonance because it has one to pass; a static panel does not, and inventing one
// would be worse than the documented beta = 1 default.
static std::pair<OpenMagnetics::Coil, std::optional<OpenMagnetics::Core>>
coil_and_core_from(const std::string& magneticOrCoilString) {
    json parsed = json::parse(magneticOrCoilString);
    if (parsed.contains("coil")) {
        // Go through Magnetic rather than constructing a Core by hand: that is the path
        // every other binding uses, so the pre-1.0 migration and the core's own
        // material/processed-description handling apply here identically instead of
        // depending on which constructor flags I happened to pick.
        OpenMagnetics::Magnetic magnetic(parsed);
        return {magnetic.get_coil(), magnetic.get_core()};
    }
    return {OpenMagnetics::Coil(parsed, false), std::nullopt};
}

std::string calculate_stray_capacitance(std::string magneticString, std::string operatingPointString, std::string modelsData){
    try {
        auto [coil, core] = coil_and_core_from(magneticString);
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto strayCapacitanceModelName = OpenMagnetics::StrayCapacitanceModels::ALBACH;
        if (models.find("strayCapacitance") != models.end()) {
            OpenMagnetics::from_json(models["strayCapacitance"], strayCapacitanceModelName);
        }

        OpenMagnetics::StrayCapacitance strayCapacitance(strayCapacitanceModelName);
        auto strayCapacitanceOutput = strayCapacitance.calculate_capacitance(coil, operatingPoint, core);

        json result;
        to_json(result, strayCapacitanceOutput);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_capacitance_matrix(std::string magneticString, std::string modelsData){
    try {
        auto [coil, core] = coil_and_core_from(magneticString);
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto strayCapacitanceModelName = OpenMagnetics::StrayCapacitanceModels::ALBACH;
        if (models.find("strayCapacitance") != models.end()) {
            OpenMagnetics::from_json(models["strayCapacitance"], strayCapacitanceModelName);
        }

        OpenMagnetics::StrayCapacitance strayCapacitance(strayCapacitanceModelName);
        auto strayCapacitanceOutput = strayCapacitance.calculate_capacitance(coil, core);

        json result;
        if (strayCapacitanceOutput.get_capacitance_matrix()) {
            auto capacitanceMatrix = strayCapacitanceOutput.get_capacitance_matrix().value();
            for (const auto& [outerKey, innerMap] : capacitanceMatrix) {
                result[outerKey] = json();
                for (const auto& [innerKey, scalarMatrix] : innerMap) {
                    json matrixJson;
                    to_json(matrixJson, scalarMatrix);
                    result[outerKey][innerKey] = matrixJson;
                }
            }
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_maxwell_capacitance_matrix(std::string magneticString, std::string modelsData){
    try {
        auto [coil, core] = coil_and_core_from(magneticString);
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto strayCapacitanceModelName = OpenMagnetics::StrayCapacitanceModels::ALBACH;
        if (models.find("strayCapacitance") != models.end()) {
            OpenMagnetics::from_json(models["strayCapacitance"], strayCapacitanceModelName);
        }

        OpenMagnetics::StrayCapacitance strayCapacitance(strayCapacitanceModelName);
        auto strayCapacitanceOutput = strayCapacitance.calculate_capacitance(coil, core);

        json result = json::array();
        if (strayCapacitanceOutput.get_maxwell_capacitance_matrix()) {
            auto maxwellMatrix = strayCapacitanceOutput.get_maxwell_capacitance_matrix().value();
            for (const auto& matrix : maxwellMatrix) {
                json matrixJson;
                to_json(matrixJson, matrix);
                result.push_back(matrixJson);
            }
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// NOTE: the former free-function binding calculate_capacitance_models_between_windings
// (energy, voltageDrop, relativeTurnsRatio) was removed. The tripole (positive 3C pi-model)
// and the canonical Biela/Kolar 6C network are now computed per winding pair inside
// StrayCapacitance::calculate_capacitance and are already serialized by
// calculate_stray_capacitance under tripoleCapacitancePerWinding / sixCapacitorNetworkPerWinding.

std::string get_available_core_losses_methods(std::string magneticString){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        auto core = magnetic.get_core();
        
        // Use CoreLossesModel::get_methods to get calculation models (IGSE, MSE, etc.)
        // instead of core.get_available_core_losses_methods() which returns data methods
        auto methods = OpenMagnetics::CoreLossesModel::get_methods(core.resolve_material());
        
        json resultJson;
        resultJson["methods"] = json::array();
        resultJson["hasMaterial"] = true;
        
        // Map CoreLossesModels to display names in preference order.
        // Keys must match the strings emitted by Definitions.h to_json for
        // CoreLossesModels (NOT the C++ enum-value names) — comparing against
        // "PROPRIETARY"/"LOSS_FACTOR" never matched, so "Loss Factor" rendered
        // as "LossFactor" via the methodKey fallback.
        std::map<std::string, int> methodPriority = {
            {"IGSE", 0},
            {"Steinmetz", 1},
            {"MSE", 2},
            {"ciGSE", 3},
            {"Roshen", 4},
            {"Barg", 5},
            {"Albach", 6},
            {"Proprietary", 7},
            {"LossFactor", 8}
        };
        
        struct MethodInfo {
            std::string key;
            std::string displayName;
            int priority;
        };
        
        std::vector<MethodInfo> methodInfos;
        
        for (const auto& method : methods) {
            json methodJson;
            to_json(methodJson, method);
            std::string methodKey = methodJson.get<std::string>();
            
            std::string displayName;
            if (methodKey == "IGSE") displayName = "IGSE";
            else if (methodKey == "Steinmetz") displayName = "Steinmetz";
            else if (methodKey == "MSE") displayName = "MSE";
            else if (methodKey == "ciGSE") displayName = "ciGSE";
            else if (methodKey == "Roshen") displayName = "Roshen";
            else if (methodKey == "Barg") displayName = "Barg";
            else if (methodKey == "Albach") displayName = "Albach";
            else if (methodKey == "Proprietary") displayName = "Proprietary";
            else if (methodKey == "LossFactor") displayName = "Loss Factor";
            else displayName = methodKey;
            
            int priority = methodPriority.count(methodKey) ? methodPriority[methodKey] : 999;
            methodInfos.push_back({methodKey, displayName, priority});
        }
        
        // Sort by priority
        std::sort(methodInfos.begin(), methodInfos.end(), 
            [](const MethodInfo& a, const MethodInfo& b) {
                return a.priority < b.priority;
            });
        
        for (const auto& info : methodInfos) {
            json methodObj;
            methodObj["key"] = info.key;
            methodObj["displayName"] = info.displayName;
            resultJson["methods"].push_back(methodObj);
        }
        
        return resultJson.dump(4);
    }
    catch (const std::exception &exc) {
        // Return empty methods array if no material or error
        json resultJson;
        resultJson["methods"] = json::array();
        resultJson["hasMaterial"] = false;
        resultJson["error"] = exc.what();
        return resultJson.dump(4);
    }
}

// Helper function to convert snake_case to Title Case
std::string toTitleCase(const std::string& str) {
    std::string result;
    bool capitalizeNext = true;
    for (char c : str) {
        if (c == '_') {
            result += ' ';
            capitalizeNext = true;
        } else if (capitalizeNext) {
            result += std::toupper(c);
            capitalizeNext = false;
        } else {
            result += std::tolower(c);
        }
    }
    return result;
}

// Helper to get all enum values as display names
std::string get_all_magnetic_field_strength_models() {
    try {
        json result = json::array();
        for (const auto& enumValue : magic_enum::enum_values<OpenMagnetics::MagneticFieldStrengthModels>()) {
            std::string name = std::string(magic_enum::enum_name(enumValue));
            result.push_back(toTitleCase(name));
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_all_fringing_effect_models() {
    try {
        json result = json::array();
        for (const auto& enumValue : magic_enum::enum_values<OpenMagnetics::MagneticFieldStrengthFringingEffectModels>()) {
            std::string name = std::string(magic_enum::enum_name(enumValue));
            result.push_back(toTitleCase(name));
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_all_reluctance_models() {
    try {
        json result = json::array();
        for (const auto& enumValue : magic_enum::enum_values<OpenMagnetics::ReluctanceModels>()) {
            std::string name = std::string(magic_enum::enum_name(enumValue));
            result.push_back(toTitleCase(name));
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_all_winding_skin_effect_models() {
    try {
        json result = json::array();
        for (const auto& enumValue : magic_enum::enum_values<OpenMagnetics::WindingSkinEffectLossesModels>()) {
            std::string name = std::string(magic_enum::enum_name(enumValue));
            result.push_back(toTitleCase(name));
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_all_winding_proximity_effect_models() {
    try {
        json result = json::array();
        for (const auto& enumValue : magic_enum::enum_values<OpenMagnetics::WindingProximityEffectLossesModels>()) {
            std::string name = std::string(magic_enum::enum_name(enumValue));
            result.push_back(toTitleCase(name));
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string get_all_stray_capacitance_models() {
    try {
        json result = json::array();
        for (const auto& enumValue : magic_enum::enum_values<OpenMagnetics::StrayCapacitanceModels>()) {
            std::string name = std::string(magic_enum::enum_name(enumValue));
            result.push_back(toTitleCase(name));
        }
        return result.dump();
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_resistance_matrix(std::string magneticString, double temperature, double frequency){
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        
        OpenMagnetics::WindingLosses windingLosses;
        auto resistanceMatrix = windingLosses.calculate_resistance_matrix(magnetic, temperature, frequency);

        json result;
        to_json(result, resistanceMatrix);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}




// Note: simulate_flyback_with_magnetic uses ngspice which is not available in WASM.
// This function is kept for PyMKF (native) usage but will throw an error in browser.

// SPICE Code Generation Functions - Returns the ngspice netlist for a converter

// Generic template for converters with turnsRatios and magnetizingInductance

// Buck SPICE generation (uses inductance directly, not turns ratios)

// Boost SPICE generation (uses inductance directly)

// SEPIC SPICE generation (uses inductanceL1 directly)

// PushPull SPICE generation

// Forward SPICE generation (Single Switch)

// Two Switch Forward SPICE generation

// Active Clamp Forward SPICE generation

// Isolated Buck SPICE generation

// Isolated Buck Boost SPICE generation

// LLC SPICE generation

// CLLC SPICE generation — uses CllcResonantParameters from
// calculate_resonant_parameters(); turnsRatio is scalar (single secondary).

// Cuk SPICE generation (inductance-shaped, mirrors generate_sepic_ngspice_circuit)

// Zeta SPICE generation (inductance-shaped)

// Four-Switch Buck-Boost SPICE generation (inductance-shaped)

// Weinberg SPICE generation (scalar turnsRatio + magnetizing inductance)

// CLLLC SPICE generation (turnsRatios vector + magnetizing inductance, mirrors SRC)

// PSHB SPICE generation (turnsRatios vector + magnetizing inductance, mirrors SRC)

// AHB SPICE generation (turnsRatios vector + magnetizing inductance, mirrors SRC)

// SRC SPICE generation — generates the ngspice netlist string (no simulation)

// DAB SPICE generation

// Phase Shifted Full Bridge SPICE generation




























// ==========================================
// Power Factor Correction (PFC) Functions
// ==========================================




// ==========================================
// Common Mode Choke (CMC) Functions
// ==========================================



// CMC SPICE Circuit Generation

// CMC LISN Test - runs ngspice with standardized CISPR test circuit

// CMC Ideal Waveforms - realistic line voltage + switching noise for design

// ==========================================
// Differential Mode Choke (DMC) Functions
// ==========================================





// DMC SPICE Circuit Generation

std::vector<size_t> get_only_temperature_dependent_indexes(std::string permeabilityPointsString) {
    try {
        std::vector<std::string> permeabilityPointsStringVector = json::parse(permeabilityPointsString);
        std::vector<PermeabilityPoint> permeabilityPoints;
        for (auto pointString : permeabilityPointsStringVector) {
            PermeabilityPoint point(json::parse(pointString));
            permeabilityPoints.push_back(point);
        }
        return OpenMagnetics::InitialPermeability::get_only_temperature_dependent_indexes(permeabilityPoints);
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return {0};
    }
}

std::vector<size_t> get_only_frequency_dependent_indexes(std::string permeabilityPointsString) {
    try {
        std::vector<std::string> permeabilityPointsStringVector = json::parse(permeabilityPointsString);
        std::vector<PermeabilityPoint> permeabilityPoints;
        for (auto pointString : permeabilityPointsStringVector) {
            PermeabilityPoint point(json::parse(pointString));
            permeabilityPoints.push_back(point);
        }
        return OpenMagnetics::InitialPermeability::get_only_frequency_dependent_indexes(permeabilityPoints);
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return {0};
    }
}

std::vector<size_t> get_only_magnetic_field_dc_bias_dependent_indexes(std::string permeabilityPointsString) {
    try {
        std::vector<std::string> permeabilityPointsStringVector = json::parse(permeabilityPointsString);
        std::vector<PermeabilityPoint> permeabilityPoints;
        for (auto pointString : permeabilityPointsStringVector) {
            PermeabilityPoint point(json::parse(pointString));
            permeabilityPoints.push_back(point);
        }
        return OpenMagnetics::InitialPermeability::get_only_magnetic_field_dc_bias_dependent_indexes(permeabilityPoints);
    }
    catch (const std::exception &exc) {
        std::cerr << std::string{exc.what()} << std::endl;
        return {0};
    }
}

std::string mas_autocomplete(std::string masString, bool simulate, std::string configurationString) {
    try {
        OpenMagnetics::Mas mas(json::parse(masString));
        json configuration(json::parse(configurationString));
        auto autocompletedMas = mas_autocomplete(mas, simulate, configuration);

        json result;
        to_json(result, autocompletedMas);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Autocomplete a bare magnetic (no inputs): what the 3D visualizer needs to enrich a design
// before MVB++ draws it (ABT #1100). The MAS variant above needs `inputs`, which a magnetic
// on its own does not carry.
std::string magnetic_autocomplete(std::string magneticString, std::string configurationString) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        json configuration(json::parse(configurationString));
        auto autocompletedMagnetic = OpenMagnetics::magnetic_autocomplete(magnetic, configuration);

        json result;
        to_json(result, autocompletedMagnetic);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_steinmetz_coefficients(std::string dataString, std::string rangesString) {
    try {
        json rangesJson = json::parse(rangesString);
        std::vector<std::pair<double, double>> ranges;
        for (auto rangeJson : rangesJson) {
            std::pair<double, double> range{rangeJson[0], rangeJson[1]};
            ranges.push_back(range);
        }
        std::vector<VolumetricLossesPoint> data(json::parse(dataString));

        auto [coefficientsPerRange, errorPerRange] = OpenMagnetics::CoreLossesSteinmetzModel::calculate_steinmetz_coefficients(data, ranges);

        // ABT #168: expose the fit error so callers can judge fit quality
        // instead of blindly trusting the coefficients.
        json coefficientsJson;
        to_json(coefficientsJson, coefficientsPerRange);
        json result;
        result["coefficientsPerRange"] = coefficientsJson;
        result["errorPerRange"] = errorPerRange;
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::map<std::string, std::string> get_initial_permeability_equations(std::string permeabilityPointString) {
    try {
        MAS::PermeabilityPoint permeabilityPoint(json::parse(permeabilityPointString));
        return OpenMagnetics::InitialPermeability::get_initial_permeability_equations(permeabilityPoint);
    }
    catch (const std::exception &exc) {
        return {{"Exception: ", std::string{exc.what()}}};
    }
}

std::map<std::string, std::string> get_core_volumetric_losses_equations(std::string coreLossesMethodDataString) {
    try {
        MAS::CoreLossesMethodData coreLossesMethodData(json::parse(coreLossesMethodDataString));
        return OpenMagnetics::CoreLossesProprietaryModel::get_core_volumetric_losses_equations(coreLossesMethodData);
    }
    catch (const std::exception &exc) {
        return {{"Exception: ", std::string{exc.what()}}};
    }
}


std::string calculate_complex_permeability(std::string coreMaterialString) {
    try {
        MAS::CoreMaterial coreMaterial(json::parse(coreMaterialString));

        ComplexPermeabilityData complexPermeabilityData;
        if (OpenMagnetics::InitialPermeability::has_frequency_dependency(coreMaterial)) {
            complexPermeabilityData = OpenMagnetics::ComplexPermeability().calculate_complex_permeability_from_frequency_dependent_initial_permeability(coreMaterial);
        }
        else {
            throw OpenMagnetics::MaterialDataMissingException("Missing complex data in material " + coreMaterial.get_name());
        }

        json result;
        to_json(result, complexPermeabilityData);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}


std::string plot_core(std::string magneticString) {
    try {
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_core(magnetic);
        painter.paint_bobbin(magnetic);
        auto result = painter.export_svg();
        return result;
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string plot_sections(std::string magneticString) {
    try {
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_core(magnetic);
        painter.paint_bobbin(magnetic);
        painter.paint_coil_sections(magnetic);
        auto result = painter.export_svg();
        return result;
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string plot_layers(std::string magneticString) {
    try {
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_core(magnetic);
        painter.paint_bobbin(magnetic);
        painter.paint_coil_layers(magnetic);
        auto result = painter.export_svg();
        return result;
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

// Wind -- or, with REAL WINDING on, re-wind -- a magnetic's coil before it is painted or its
// connection layout is exported.
//
// ABT #849 (Alf, 2026-08-22): with real winding on, wind IN-PROCESS even when the MAS already
// carries turns. `_realWindingBlockingApplied` is a runtime member of Coil, set inside wind() and
// never serialized into MAS, so a coil deserialized from JSON always reports "blocking not
// applied" -- and paint_coil_connections then paints every terminal and link as a RED DASHED
// "declined" marker, which is what made the web view disagree with the same design's locally
// painted SVG. Re-winding makes the flag state the truth; a genuine ABT #650 decline still shows.
//
// The gate and the wind are MKF's own (magnetic_coil_needs_winding / wind_magnetic_coil_as_described,
// the path magnetic_autocomplete takes), so the re-wind keeps what the design says about itself:
// the interleaving its stored sections describe (ABT #610) and the turns alignment autocomplete
// used. A bare coil.wind() here re-wound with the default [0,1] x interleavingLevel pattern, so a
// S,P,S,P,S,P,S design was painted as two plain sections -- a coil that is not the design.
//
// A wind that does not fit is still DRAWN (Alf, 2026-09-29): "it does not fit" alone does not
// tell the user why, the drawing does. The painters draw the coil as wound, add MKF's fit overlay
// (window outlined, escaping turns ringed in red) and embed the reason in the SVG as
// <desc id="om-winding-fit-failure">, which the web shows beside the drawing. The reason is
// returned here; std::nullopt means the coil fits (or was not re-wound).
static std::optional<std::string> wind_coil_for_display(OpenMagnetics::Magnetic& magnetic, const std::string& caller) {
    if (!OpenMagnetics::magnetic_coil_needs_winding(magnetic)) {
        return std::nullopt;
    }
    if (OpenMagnetics::wind_magnetic_coil_as_described(magnetic)) {
        return std::nullopt;
    }
    const auto& reason = magnetic.get_mutable_coil().get_last_fit_failure();
    return caller + ": the coil does not fit its winding window when wound as the design describes it. " +
           (reason.empty() ? std::string("MKF could not narrow the reason further.") : reason);
}

// Carries a fit failure inside the drawing: an SVG <desc> right after the opening <svg ...> tag.
static std::string embed_winding_fit_failure(const std::string& svg, const std::optional<std::string>& fitFailure) {
    if (!fitFailure) {
        return svg;
    }
    const auto openTagEnd = svg.find('>', svg.find("<svg"));
    if (svg.rfind("<svg", 0) != 0 || openTagEnd == std::string::npos) {
        throw std::runtime_error("embed_winding_fit_failure: the painter did not return an SVG to carry the fit failure in");
    }
    std::string escaped;
    for (char c : fitFailure.value()) {
        switch (c) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            default: escaped += c;
        }
    }
    return svg.substr(0, openTagEnd + 1) + "<desc id=\"om-winding-fit-failure\">" + escaped + "</desc>" +
           svg.substr(openTagEnd + 1);
}

std::string plot_turns(std::string magneticString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        
        auto magneticJson = json::parse(magneticString);
        
        OpenMagnetics::Magnetic magnetic(magneticJson);

        // Ensure the coil is wound; otherwise paint_coil_turns throws COIL_NOT_PROCESSED.
        // Catalog magnetics often arrive with only functionalDescription populated. With real
        // winding on it is re-wound in-process (ABT #849), see wind_coil_for_display.
        const auto fitFailure = wind_coil_for_display(magnetic, "plot_turns");

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_core(magnetic);
        painter.paint_bobbin(magnetic);
        painter.paint_coil_turns(magnetic);
        if (fitFailure) {
            painter.paint_winding_fit_problems(magnetic);
        }
        return embed_winding_fit_failure(painter.export_svg(), fitFailure);
    }
    catch(const std::exception& ex)
    {
        // "Exception: " is what the web's export paths test for before saving the result as an
        // .svg; a bare what() slipped past them and was downloaded as the drawing.
        return "Exception: " + std::string{ex.what()};
    }
    catch(...)
    {
        return "Exception: Unknown failure occurred. Possible memory corruption";
    }
}


// REAL WINDING (ABT #646 / #631 follow-up). plot_turns draws core + bobbin + turns and stops
// there — it never draws how the turns are CONNECTED. paint_magnetic is the entry point that
// does: XY adds the inter-layer links, dragbacks and terminal leads on top of the usual front
// view; YZ is the connection-face projection where those leads are seen end-on.
//
// projection: "XY" (default) or "YZ", case-insensitive.
//
// The coil is (re-)wound when it has no turnsDescription, exactly like plot_turns. The caller
// decides whether real-winding blocking applies by setting coilUseRealWindingGeometry through
// set_settings first: that flag changes the LAYOUT (MKF reserves the lead and dragback
// corridors), so painting with it off would draw a different coil from the one the 3D builder
// routes. One flag, one coil, both views.
// Drain MKF's capture log to JS. Registering the sink also raises the logger to WARNING, so
// call it once BEFORE the operation you want to observe, then again to read what it emitted.
std::string read_log() {
    return OpenMagnetics::read_log();
}

std::string plot_magnetic(std::string magneticString, std::string projectionString) {
    try {
        std::filesystem::path emptyFilepath;

        auto magneticJson = json::parse(magneticString);
        OpenMagnetics::Magnetic magnetic(magneticJson);

        // Wound when it has no turns, and re-wound in-process with real winding on (ABT #849),
        // see wind_coil_for_display.
        const auto fitFailure = wind_coil_for_display(magnetic, "plot_magnetic");

        std::string projection = projectionString;
        std::transform(projection.begin(), projection.end(), projection.begin(),
                       [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        OpenMagnetics::PainterProjection painterProjection;
        if (projection == "YZ") {
            painterProjection = OpenMagnetics::PainterProjection::YZ;
        }
        else if (projection.empty() || projection == "XY") {
            painterProjection = OpenMagnetics::PainterProjection::XY;
        }
        else {
            // Named explicitly rather than silently falling back to XY: a caller asking for a
            // projection that does not exist is a bug in the caller, and a view that quietly
            // is not the one requested is the worst way to find out.
            throw std::runtime_error("plot_magnetic: unknown projection '" + projectionString +
                                     "', expected 'XY' or 'YZ'");
        }

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_magnetic(magnetic, painterProjection);
        if (fitFailure && painterProjection == OpenMagnetics::PainterProjection::XY) {
            // The overlay is drawn in the XY frame; the YZ face carries the reason text only.
            painter.paint_winding_fit_problems(magnetic);
        }
        return embed_winding_fit_failure(painter.export_svg(), fitFailure);
    }
    catch(const std::exception& ex)
    {
        // "Exception: " is what the web's export paths test for before saving the result as an
        // .svg; a bare what() slipped past them and was downloaded as the drawing.
        return "Exception: " + std::string{ex.what()};
    }
    catch(...)
    {
        return "Exception: Unknown failure occurred. Possible memory corruption";
    }
}


// ABT #849 (Alf, 2026-08-22): THE CONNECTION LAYOUT AS DATA, for the Coil Studio.
// The Studio is pure JS/SVG built from the MAS, while connections are DERIVED in C++
// (Coil::get_connection_reserved_spaces / get_connection_layout) and are deliberately not part
// of the MAS contract -- they would go stale the moment anything is re-wound. So expose them:
// one call returns every drawn marker, the routes they group into, and the ride levels those
// routes impose, in the same winding-window frame the turns already use. The Studio draws them;
// MKF stays the single source of truth for where they go.
std::string get_connection_layout(std::string magneticString) {
    try {
        auto magneticJson = json::parse(magneticString);
        OpenMagnetics::Magnetic magnetic(magneticJson);
        auto& settings = OpenMagnetics::Settings::GetInstance();
        const bool realWinding = settings.get_coil_use_real_winding_geometry();

        // WIND IN-PROCESS whenever real winding is on. `_realWindingBlockingApplied` is a RUNTIME
        // member of Coil (set inside wind()), never serialized into MAS, so a coil deserialized
        // from JSON always reports "blocking not applied" however it was really wound -- which is
        // what painted every web connection as a red dashed "declined" marker while the local
        // (same-process) SVGs drew them filled. Winding here is deterministic -- same sections and
        // layers in, same layout out -- so the geometry is the one the app already stored, and the
        // flag then states the truth: when the ABT #650 gate really declines, consumers must show
        // it rather than present an unpaid-for layout as real.
        // Same gate and same wind as the painters (wind_coil_for_display): the stored sections
        // pattern is honoured, and a wind that does not fit throws instead of exporting a
        // connection layout for a coil the design does not have.
        if (auto fitFailure = wind_coil_for_display(magnetic, "get_connection_layout")) {
            // Data, not a drawing: there is nothing to show the problem on, so it stays an error.
            throw std::runtime_error(fitFailure.value());
        }
        auto& coil = magnetic.get_mutable_coil();

        auto enumName = [](auto value) { return std::string(magic_enum::enum_name(value)); };

        json out;
        out["realWinding"] = realWinding;
        out["blockingApplied"] = coil.is_real_winding_blocking_applied();

        json markers = json::array();
        for (const auto& space : coil.get_connection_reserved_spaces()) {
            json m;
            m["winding"] = space.winding;
            m["parallel"] = space.parallel;
            m["section"] = space.section;
            m["layer"] = space.layer;                  // non-empty = a squeeze on a CROSSED layer
            m["coordinates"] = space.coordinates;      // centre, cartesian as turns are
            m["dimensions"] = space.dimensions;
            m["rotation"] = space.rotation;            // degrees; non-zero on diagonal links
            m["coordinateSystem"] = enumName(space.coordinateSystem);
            m["plane"] = enumName(space.plane);        // WINDOW_XY or FRONT_YZ (out of this view)
            m["kind"] = enumName(space.kind);
            m["isTerminal"] = space.isTerminal;
            m["edgeDepth"] = space.edgeDepth;
            m["fromTurn"] = space.fromTurn;
            m["toTurn"] = space.toTurn;
            if (space.routedLength) {
                m["routedLength"] = space.routedLength.value();
            }
            markers.push_back(m);
        }
        out["markers"] = markers;

        auto layout = coil.get_connection_layout();
        json routes = json::array();
        for (const auto& route : layout.routes) {
            json r;
            r["winding"] = route.winding;
            r["parallel"] = route.parallel;
            r["fromTurn"] = route.fromTurn;
            r["toTurn"] = route.toTurn;
            r["kind"] = enumName(route.kind);
            r["side"] = route.side;
            r["waypoints"] = route.waypoints;          // {layerAxis, turnAxis}, in path order
            r["routedLength"] = route.routedLength;
            routes.push_back(r);
        }
        out["routes"] = routes;

        json rideLevels = json::array();
        for (const auto& level : layout.rideLevels) {
            json l;
            l["side"] = level.side;
            l["radius"] = level.radius;
            l["height"] = level.height;
            rideLevels.push_back(l);
        }
        out["rideLevels"] = rideLevels;

        return out.dump();
    }
    catch(const std::runtime_error& re)
    {
        return json{{"error", re.what()}}.dump();
    }
    catch(const std::exception& ex)
    {
        return json{{"error", ex.what()}}.dump();
    }
    catch(...)
    {
        return json{{"error", "Unknown failure occurred. Possible memory corruption"}}.dump();
    }
}

std::string plot_magnetic_field(std::string magneticString, std::string operatingPointString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        // For toroidal cores, ensure the coil is wound to generate additional_coordinates.
        // Wound as the design describes it (stored sections pattern kept); a wind that does not
        // fit is still drawn, with the reason carried in the SVG -- see wind_coil_for_display.
        std::optional<std::string> fitFailure;
        if (magnetic.get_mutable_core().get_shape_family() == CoreShapeFamily::T &&
            (!magnetic.get_mutable_coil().get_turns_description() ||
             magnetic.get_mutable_coil().get_turns_description()->empty())) {
            fitFailure = wind_coil_for_display(magnetic, "plot_magnetic_field");
        }

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_magnetic_field(operatingPoint, magnetic);
        painter.paint_core(magnetic);
        // painter.paint_bobbin(magnetic);
        // Paint turns for H field, skip insulation tape and margin for cleaner visualization
        painter.paint_coil_turns(magnetic, true);
        return embed_winding_fit_failure(painter.export_svg(), fitFailure);
    }
    catch(const std::exception& ex)
    {
        // "Exception: " is what the web's export paths test for before saving the result as an
        // .svg; a bare what() slipped past them and was downloaded as the drawing.
        return "Exception: " + std::string{ex.what()};
    }
    catch(...)
    {
        return "Exception: Unknown failure occurred. Possible memory corruption";
    }
}


std::string plot_electric_field(std::string magneticString, std::string operatingPointString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        // For toroidal cores, ensure the coil is wound to generate additional_coordinates.
        // Wound as the design describes it (stored sections pattern kept); a wind that does not
        // fit is still drawn, with the reason carried in the SVG -- see wind_coil_for_display.
        std::optional<std::string> fitFailure;
        if (magnetic.get_mutable_core().get_shape_family() == CoreShapeFamily::T &&
            (!magnetic.get_mutable_coil().get_turns_description() ||
             magnetic.get_mutable_coil().get_turns_description()->empty())) {
            fitFailure = wind_coil_for_display(magnetic, "plot_electric_field");
        }

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_electric_field(operatingPoint, magnetic);
        painter.paint_core(magnetic);
        // painter.paint_bobbin(magnetic);
        // Paint turns for E field, skip insulation tape and margin for cleaner visualization
        painter.paint_coil_turns(magnetic, true);
        return embed_winding_fit_failure(painter.export_svg(), fitFailure);
    }
    catch(const std::exception& ex)
    {
        // "Exception: " is what the web's export paths test for before saving the result as an
        // .svg; a bare what() slipped past them and was downloaded as the drawing.
        return "Exception: " + std::string{ex.what()};
    }
    catch(...)
    {
        return "Exception: Unknown failure occurred. Possible memory corruption";
    }
}


std::string plot_wire_losses(std::string magneticString, std::string operatingPointString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_core(magnetic);
        painter.paint_bobbin(magnetic);
        // Paint turns with margins/layers
        painter.paint_coil_turns(magnetic);
        // Then paint wire losses on top
        try {
            painter.paint_wire_losses(magnetic, std::nullopt, operatingPoint);
        } catch (const std::exception& e) {
            // Wire losses already painted via coil_turns fallback
        }
        auto result = painter.export_svg();
        return result;
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string plot_wire(std::string wireString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(false);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(true);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Wire wire(json::parse(wireString));
        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_wire(wire);
        auto result = painter.export_svg();
        return result;
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string plot_temperature_field(std::string magneticString, std::string operatingPointString, std::string textColor = "#000000", std::string bgColor = "#FFFFFF") {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        // For toroidal cores, ensure the coil is wound -- as the design describes it; a wind that
        // does not fit is still drawn, with the reason carried in the SVG (see wind_coil_for_display).
        std::optional<std::string> fitFailure;
        if (magnetic.get_mutable_core().get_shape_family() == CoreShapeFamily::T &&
            (!magnetic.get_mutable_coil().get_turns_description() ||
             magnetic.get_mutable_coil().get_turns_description()->empty())) {
            fitFailure = wind_coil_for_display(magnetic, "plot_temperature_field");
        }
        
        // Get ambient temperature from operating point
        double ambientTemperature = operatingPoint.get_conditions().get_ambient_temperature();
        
        // Run magnetic simulation to get losses
        OpenMagnetics::MagneticSimulator magneticSimulator;
        OpenMagnetics::Mas mas;
        mas.set_magnetic(magnetic);
        mas.get_mutable_inputs().set_operating_points({operatingPoint});  // Set the operating point for simulation
        auto simulatedMas = magneticSimulator.simulate(mas);
        if (simulatedMas.get_outputs().empty()) {
            throw std::runtime_error("plot_temperature_field: simulation produced no outputs");
        }

        // Shared config builder (ABT #906): the same configuration MagneticSimulator uses
        // for outputs[].temperature, so this plot and the exported MAS can never disagree.
        auto config = OpenMagnetics::TemperatureConfig::fromSimulatedOutput(operatingPoint, simulatedMas.get_outputs()[0]);

        // Create temperature model and calculate temperatures
        OpenMagnetics::Temperature temperature(magnetic, config);
        auto thermalResult = temperature.calculateTemperatures();
        
        // Use Painter class (same as magnetic field)
        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_temperature_field(magnetic, thermalResult.nodeTemperatures, true, OpenMagnetics::ColorPalette::BLUE_TO_RED, ambientTemperature, textColor, bgColor);
        // Note: paint_core and paint_coil_turns are NOT called here because they would draw
        // the standard ferrite/copper geometry on top of the temperature visualization,
        // hiding the temperature colors. The temperature field function already draws
        // the core and turns with their temperature colors.
        
        return embed_winding_fit_failure(painter.export_svg(), fitFailure);
    }
    catch(const std::exception& ex)
    {
        // "Exception: " is what the web's export paths test for before saving the result as an
        // .svg; a bare what() slipped past them and was downloaded as the drawing.
        return "Exception: " + std::string{ex.what()};
    }
    catch(...)
    {
        return "Exception: Unknown failure occurred. Possible memory corruption";
    }
}


std::string set_interlayer_insulation(std::string coilString, double layerThickness){
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        coil.set_interlayer_insulation(layerThickness);

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string set_intersection_insulation(std::string coilString, double layerThickness, int numberInsulationLayers){
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        coil.set_intersection_insulation(layerThickness, numberInsulationLayers);

        json result;
        to_json(result, coil);
        return result.dump(4);
    }
    catch(const std::runtime_error& re)
    {
        return re.what();
    }
    catch(const std::exception& ex)
    {
        return ex.what();
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }
}

std::string calculate_filling_factor(std::string coilString) {
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        auto fillingFactors = coil.calculate_filling_factor();
        json result;
        // ABT #245: areaFillingFactor is the TRUE area fraction. The overfill that used
        // to be folded into it is reported separately as maxLayerFillingFactor, and
        // windingFits carries the "does this wind at all" verdict the UI needs — a coil
        // with a degenerate section used to report 6077% area fill for a 2.35% winding.
        result["areaFillingFactor"] = fillingFactors.areaFillingFactor;
        result["maxLayerFillingFactor"] = fillingFactors.maxLayerFillingFactor;
        result["overlappingFillingFactor"] = fillingFactors.overlappingFillingFactor;
        result["contiguousFillingFactor"] = fillingFactors.contiguousFillingFactor;
        result["windingFits"] = fillingFactors.windingFits;
        return result.dump(4);
    }
    // "Exception: " prefix, as 109 other bindings do. The JS side keys on that prefix to
    // tell an error from a result; returning a bare what() made taskQueue.js JSON.parse
    // the message, so a std::bad_optional_access surfaced as
    //   SyntaxError: Unexpected token 'b', "bad_optional_access" is not valid JSON
    // — the one string that says nothing about what failed or where.
    catch(const std::runtime_error& re)
    {
        return "Exception: " + std::string(re.what());
    }
    catch(const std::exception& ex)
    {
        return "Exception: " + std::string(ex.what());
    }
    catch(...)
    {
        return "Unknown failure occurred. Possible memory corruption";
    }

}

std::string get_settings() {
    try {
        json settingsJson;

        settingsJson["magnetizingInductanceIncludeAirInductance"] = OpenMagnetics::Settings::GetInstance().get_magnetizing_inductance_include_air_inductance();
        settingsJson["coilAllowMarginTape"] = OpenMagnetics::Settings::GetInstance().get_coil_allow_margin_tape();
        settingsJson["coilAllowInsulatedWire"] = OpenMagnetics::Settings::GetInstance().get_coil_allow_insulated_wire();
        settingsJson["coilFillSectionsWithMarginTape"] = OpenMagnetics::Settings::GetInstance().get_coil_fill_sections_with_margin_tape();
        settingsJson["coilWindEvenIfNotFit"] = OpenMagnetics::Settings::GetInstance().get_coil_wind_even_if_not_fit();
        settingsJson["coilDelimitAndCompact"] = OpenMagnetics::Settings::GetInstance().get_coil_delimit_and_compact();
        settingsJson["coilOnlyOneTurnPerLayerInContiguousRectangular"] = OpenMagnetics::Settings::GetInstance().get_coil_only_one_turn_per_layer_in_contiguous_rectangular();
        settingsJson["coilTryRewind"] = OpenMagnetics::Settings::GetInstance().get_coil_try_rewind();
        settingsJson["coilMaximumLayersPlanar"] = OpenMagnetics::Settings::GetInstance().get_coil_maximum_layers_planar();
        // ABT #1099: the manufacturer whose materials the core adviser searches first
        // (standard-cores mode; a tiebreak, not a gate — see CoreAdviserMaterials).
        settingsJson["preferredCoreMaterialFerriteManufacturer"] = OpenMagnetics::Settings::GetInstance().get_preferred_core_material_ferrite_manufacturer();
        settingsJson["preferredCoreMaterialPowderManufacturer"] = OpenMagnetics::Settings::GetInstance().get_preferred_core_material_powder_manufacturer();
        // ABT #1110: wire standard the wire/coil advisers restrict to (null = no preference).
        {
            auto preferredWireStandard = OpenMagnetics::Settings::GetInstance().get_preferred_wire_standard();
            if (preferredWireStandard) {
                json preferredWireStandardJson;
                to_json(preferredWireStandardJson, preferredWireStandard.value());
                settingsJson["preferredWireStandard"] = preferredWireStandardJson;
            }
            else {
                settingsJson["preferredWireStandard"] = nullptr;
            }
        }
        settingsJson["coilIncludeAdditionalCoordinates"] = OpenMagnetics::Settings::GetInstance().get_coil_include_additional_coordinates();
        // Real winding: MKF lays the turns out as they are actually wound (leads, pitch,
        // dragbacks) instead of the idealised per-turn rings. The web 2D view paints
        // through plot_turns, which reads this setting, so without it on the round trip
        // the frontend has no way to ask for the real winding in 2D at all.
        settingsJson["coilUseRealWindingGeometry"] = OpenMagnetics::Settings::GetInstance().get_coil_use_real_winding_geometry();

        settingsJson["useOnlyCoresInStock"] = OpenMagnetics::Settings::GetInstance().get_use_only_cores_in_stock();
        settingsJson["painterNumberPointsX"] = OpenMagnetics::Settings::GetInstance().get_painter_number_points_x();
        settingsJson["painterNumberPointsY"] = OpenMagnetics::Settings::GetInstance().get_painter_number_points_y();
        settingsJson["painterMirroringDimension"] = OpenMagnetics::Settings::GetInstance().get_painter_mirroring_dimension();
        settingsJson["painterMode"] = OpenMagnetics::Settings::GetInstance().get_painter_mode();
        settingsJson["painterLogarithmicScale"] = OpenMagnetics::Settings::GetInstance().get_painter_logarithmic_scale();
        settingsJson["painterIncludeFringing"] = OpenMagnetics::Settings::GetInstance().get_painter_include_fringing();
        if (OpenMagnetics::Settings::GetInstance().get_painter_maximum_value_colorbar()) {
            settingsJson["painterMaximumValueColorbar"] = OpenMagnetics::Settings::GetInstance().get_painter_maximum_value_colorbar();
        }
        if (OpenMagnetics::Settings::GetInstance().get_painter_minimum_value_colorbar()) {
            settingsJson["painterMinimumValueColorbar"] = OpenMagnetics::Settings::GetInstance().get_painter_minimum_value_colorbar();
        }
        settingsJson["painterColorFerrite"] = OpenMagnetics::Settings::GetInstance().get_painter_color_ferrite();
        settingsJson["painterColorBobbin"] = OpenMagnetics::Settings::GetInstance().get_painter_color_bobbin();
        settingsJson["painterColorCopper"] = OpenMagnetics::Settings::GetInstance().get_painter_color_copper();
        settingsJson["painterColorInsulation"] = OpenMagnetics::Settings::GetInstance().get_painter_color_insulation();
        settingsJson["painterColorMargin"] = OpenMagnetics::Settings::GetInstance().get_painter_color_margin();
        settingsJson["painterColorSpacer"] = OpenMagnetics::Settings::GetInstance().get_painter_color_spacer();
        settingsJson["painterDrawSpacer"] = OpenMagnetics::Settings::GetInstance().get_painter_draw_spacer();
        settingsJson["magneticFieldNumberPointsX"] = OpenMagnetics::Settings::GetInstance().get_magnetic_field_number_points_x();
        settingsJson["magneticFieldNumberPointsY"] = OpenMagnetics::Settings::GetInstance().get_magnetic_field_number_points_y();
        settingsJson["magneticFieldMirroringDimension"] = OpenMagnetics::Settings::GetInstance().get_magnetic_field_mirroring_dimension();
        settingsJson["magneticFieldIncludeFringing"] = OpenMagnetics::Settings::GetInstance().get_magnetic_field_include_fringing();
        settingsJson["coilAdviserMaximumNumberWires"] = OpenMagnetics::Settings::GetInstance().get_coil_adviser_maximum_number_wires();
        settingsJson["coreIncludeMargin"] = OpenMagnetics::Settings::GetInstance().get_core_adviser_include_margin();
        settingsJson["coreIncludeStacks"] = OpenMagnetics::Settings::GetInstance().get_core_adviser_include_stacks();
        settingsJson["coreIncludeDistributedGaps"] = OpenMagnetics::Settings::GetInstance().get_core_adviser_include_distributed_gaps();
        settingsJson["verbose"] = OpenMagnetics::Settings::GetInstance().get_verbose();

        settingsJson["useToroidalCores"] = OpenMagnetics::Settings::GetInstance().get_use_toroidal_cores();
        settingsJson["useConcentricCores"] = OpenMagnetics::Settings::GetInstance().get_use_concentric_cores();

        // Multi-column winding placement (winding studio)
        settingsJson["corePerColumnWindingWindows"] = OpenMagnetics::Settings::GetInstance().get_core_per_column_winding_windows();
        settingsJson["coilAdviserAllowLateralPlacement"] = OpenMagnetics::Settings::GetInstance().get_coil_adviser_allow_lateral_placement();

        // Temperature-filter settings were removed from Settings upstream;
        // emit defaults so the frontend schema isn't broken.
        settingsJson["coreAdviserEnableTemperatureFilter"] = false;
        settingsJson["coreAdviserMaximumTemperature"] = 130.0;

        // Model selection settings
        settingsJson["magneticFieldStrengthModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_magnetic_field_strength_model());
        settingsJson["magneticFieldStrengthFringingEffectModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_magnetic_field_strength_fringing_effect_model());
        settingsJson["reluctanceModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_reluctance_model());
        settingsJson["coreTemperatureModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_core_temperature_model());
        settingsJson["coreThermalResistanceModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_core_thermal_resistance_model());
        settingsJson["windingSkinEffectLossesModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_winding_skin_effect_losses_model());
        settingsJson["windingProximityEffectLossesModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_winding_proximity_effect_losses_model());
        settingsJson["strayCapacitanceModel"] = static_cast<int>(OpenMagnetics::Settings::GetInstance().get_stray_capacitance_model());
        settingsJson["coilEnableUserWindingLossesModels"] = OpenMagnetics::Settings::GetInstance().get_coil_enable_user_winding_losses_models();

        // ABT #1454: a turn overlapping its enclosure makes the thermal network throw (true,
        // the default) or lose that face's conduction path with an ERROR logged (false).
        settingsJson["thermalNetworkStrictGeometry"] = OpenMagnetics::Settings::GetInstance().get_thermal_network_strict_geometry();

        return settingsJson.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

void set_settings(std::string settingsString) {
    json settingsJson = json::parse(settingsString);

    OpenMagnetics::Settings::GetInstance().set_magnetizing_inductance_include_air_inductance(settingsJson["magnetizingInductanceIncludeAirInductance"]);
    OpenMagnetics::Settings::GetInstance().set_coil_allow_margin_tape(settingsJson["coilAllowMarginTape"]);
    OpenMagnetics::Settings::GetInstance().set_coil_allow_insulated_wire(settingsJson["coilAllowInsulatedWire"]);
    OpenMagnetics::Settings::GetInstance().set_coil_fill_sections_with_margin_tape(settingsJson["coilFillSectionsWithMarginTape"]);
    OpenMagnetics::Settings::GetInstance().set_coil_wind_even_if_not_fit(settingsJson["coilWindEvenIfNotFit"]);
    OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(settingsJson["coilDelimitAndCompact"]);
    OpenMagnetics::Settings::GetInstance().set_coil_only_one_turn_per_layer_in_contiguous_rectangular(settingsJson["coilOnlyOneTurnPerLayerInContiguousRectangular"]);
    OpenMagnetics::Settings::GetInstance().set_coil_try_rewind(settingsJson["coilTryRewind"]);
    OpenMagnetics::Settings::GetInstance().set_coil_maximum_layers_planar(settingsJson["coilMaximumLayersPlanar"]);
    if (settingsJson.contains("preferredCoreMaterialFerriteManufacturer")) {
        OpenMagnetics::Settings::GetInstance().set_preferred_core_material_ferrite_manufacturer(settingsJson["preferredCoreMaterialFerriteManufacturer"]);
    }
    if (settingsJson.contains("preferredCoreMaterialPowderManufacturer")) {
        OpenMagnetics::Settings::GetInstance().set_preferred_core_material_powder_manufacturer(settingsJson["preferredCoreMaterialPowderManufacturer"]);
    }
    if (settingsJson.contains("preferredWireStandard")) {
        if (settingsJson["preferredWireStandard"].is_null()) {
            OpenMagnetics::Settings::GetInstance().set_preferred_wire_standard(std::nullopt);
        }
        else {
            OpenMagnetics::Settings::GetInstance().set_preferred_wire_standard(settingsJson["preferredWireStandard"].get<WireStandard>());
        }
    }
    if (settingsJson.contains("coilIncludeAdditionalCoordinates")) {
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(settingsJson["coilIncludeAdditionalCoordinates"]);
    }
    // Guarded like the key above: older callers that hand back a settings object from
    // before this key existed must keep working rather than throw on a missing field.
    if (settingsJson.contains("coilUseRealWindingGeometry")) {
        OpenMagnetics::Settings::GetInstance().set_coil_use_real_winding_geometry(settingsJson["coilUseRealWindingGeometry"]);
    }

    OpenMagnetics::Settings::GetInstance().set_use_only_cores_in_stock(settingsJson["useOnlyCoresInStock"]);
    OpenMagnetics::Settings::GetInstance().set_painter_number_points_x(settingsJson["painterNumberPointsX"]);
    OpenMagnetics::Settings::GetInstance().set_painter_number_points_y(settingsJson["painterNumberPointsY"]);
    OpenMagnetics::Settings::GetInstance().set_painter_mirroring_dimension(settingsJson["painterMirroringDimension"]);
    OpenMagnetics::Settings::GetInstance().set_painter_mode(settingsJson["painterMode"]);
    OpenMagnetics::Settings::GetInstance().set_painter_logarithmic_scale(settingsJson["painterLogarithmicScale"]);
    OpenMagnetics::Settings::GetInstance().set_painter_include_fringing(settingsJson["painterIncludeFringing"]);
    if (settingsJson.contains("painterMaximumValueColorbar")) {
        OpenMagnetics::Settings::GetInstance().set_painter_maximum_value_colorbar(settingsJson["painterMaximumValueColorbar"]);
    }
    if (settingsJson.contains("painterMinimumValueColorbar")) {
        OpenMagnetics::Settings::GetInstance().set_painter_minimum_value_colorbar(settingsJson["painterMinimumValueColorbar"]);
    }
    OpenMagnetics::Settings::GetInstance().set_painter_color_ferrite(settingsJson["painterColorFerrite"]);
    OpenMagnetics::Settings::GetInstance().set_painter_color_bobbin(settingsJson["painterColorBobbin"]);
    OpenMagnetics::Settings::GetInstance().set_painter_color_copper(settingsJson["painterColorCopper"]);
    OpenMagnetics::Settings::GetInstance().set_painter_color_insulation(settingsJson["painterColorInsulation"]);
    OpenMagnetics::Settings::GetInstance().set_painter_color_margin(settingsJson["painterColorMargin"]);
    if (settingsJson.contains("painterColorSpacer")) {
        OpenMagnetics::Settings::GetInstance().set_painter_color_spacer(settingsJson["painterColorSpacer"]);
    }
    if (settingsJson.contains("painterDrawSpacer")) {
        OpenMagnetics::Settings::GetInstance().set_painter_draw_spacer(settingsJson["painterDrawSpacer"]);
    }
    OpenMagnetics::Settings::GetInstance().set_magnetic_field_number_points_x(settingsJson["magneticFieldNumberPointsX"]);
    OpenMagnetics::Settings::GetInstance().set_magnetic_field_number_points_y(settingsJson["magneticFieldNumberPointsY"]);
    OpenMagnetics::Settings::GetInstance().set_magnetic_field_mirroring_dimension(settingsJson["magneticFieldMirroringDimension"]);
    OpenMagnetics::Settings::GetInstance().set_magnetic_field_include_fringing(settingsJson["magneticFieldIncludeFringing"]);
    OpenMagnetics::Settings::GetInstance().set_coil_adviser_maximum_number_wires(settingsJson["coilAdviserMaximumNumberWires"]);
    OpenMagnetics::Settings::GetInstance().set_core_adviser_include_margin(settingsJson["coreIncludeMargin"]);
    OpenMagnetics::Settings::GetInstance().set_core_adviser_include_stacks(settingsJson["coreIncludeStacks"]);
    OpenMagnetics::Settings::GetInstance().set_core_adviser_include_distributed_gaps(settingsJson["coreIncludeDistributedGaps"]);
    OpenMagnetics::Settings::GetInstance().set_verbose(settingsJson["verbose"]);

    OpenMagnetics::Settings::GetInstance().set_use_toroidal_cores(settingsJson["useToroidalCores"]);
    OpenMagnetics::Settings::GetInstance().set_use_concentric_cores(settingsJson["useConcentricCores"]);

    // coreAdviserEnableTemperatureFilter / coreAdviserMaximumTemperature:
    // setters were removed from Settings upstream; accept the keys for
    // forward-compat but don't persist — no-op.

    // Model selection settings
    if (settingsJson.contains("magneticFieldStrengthModel")) {
        OpenMagnetics::Settings::GetInstance().set_magnetic_field_strength_model(static_cast<OpenMagnetics::MagneticFieldStrengthModels>(settingsJson["magneticFieldStrengthModel"].get<int>()));
    }
    if (settingsJson.contains("magneticFieldStrengthFringingEffectModel")) {
        OpenMagnetics::Settings::GetInstance().set_magnetic_field_strength_fringing_effect_model(static_cast<OpenMagnetics::MagneticFieldStrengthFringingEffectModels>(settingsJson["magneticFieldStrengthFringingEffectModel"].get<int>()));
    }
    if (settingsJson.contains("reluctanceModel")) {
        OpenMagnetics::Settings::GetInstance().set_reluctance_model(static_cast<OpenMagnetics::ReluctanceModels>(settingsJson["reluctanceModel"].get<int>()));
    }
    if (settingsJson.contains("coreTemperatureModel")) {
        OpenMagnetics::Settings::GetInstance().set_core_temperature_model(static_cast<OpenMagnetics::CoreTemperatureModels>(settingsJson["coreTemperatureModel"].get<int>()));
    }
    if (settingsJson.contains("coreThermalResistanceModel")) {
        OpenMagnetics::Settings::GetInstance().set_core_thermal_resistance_model(static_cast<OpenMagnetics::CoreThermalResistanceModels>(settingsJson["coreThermalResistanceModel"].get<int>()));
    }
    if (settingsJson.contains("windingSkinEffectLossesModel")) {
        OpenMagnetics::Settings::GetInstance().set_winding_skin_effect_losses_model(static_cast<OpenMagnetics::WindingSkinEffectLossesModels>(settingsJson["windingSkinEffectLossesModel"].get<int>()));
    }
    if (settingsJson.contains("windingProximityEffectLossesModel")) {
        OpenMagnetics::Settings::GetInstance().set_winding_proximity_effect_losses_model(static_cast<OpenMagnetics::WindingProximityEffectLossesModels>(settingsJson["windingProximityEffectLossesModel"].get<int>()));
    }
    if (settingsJson.contains("strayCapacitanceModel")) {
        OpenMagnetics::Settings::GetInstance().set_stray_capacitance_model(static_cast<OpenMagnetics::StrayCapacitanceModels>(settingsJson["strayCapacitanceModel"].get<int>()));
    }
    if (settingsJson.contains("coilEnableUserWindingLossesModels")) {
        bool value = settingsJson["coilEnableUserWindingLossesModels"].get<bool>();
        OpenMagnetics::Settings::GetInstance().set_coil_enable_user_winding_losses_models(value);
    }

    // Multi-column winding placement (winding studio). Guarded: older persisted
    // settings objects predate these keys.
    if (settingsJson.contains("corePerColumnWindingWindows")) {
        OpenMagnetics::Settings::GetInstance().set_core_per_column_winding_windows(settingsJson["corePerColumnWindingWindows"].get<bool>());
    }
    if (settingsJson.contains("coilAdviserAllowLateralPlacement")) {
        OpenMagnetics::Settings::GetInstance().set_coil_adviser_allow_lateral_placement(settingsJson["coilAdviserAllowLateralPlacement"].get<bool>());
    }
    // Guarded: settings objects persisted before this key existed must keep working.
    if (settingsJson.contains("thermalNetworkStrictGeometry")) {
        OpenMagnetics::Settings::GetInstance().set_thermal_network_strict_geometry(settingsJson["thermalNetworkStrictGeometry"].get<bool>());
    }
}
void reset_settings(std::string settingsString) {
    OpenMagnetics::Settings::GetInstance().reset();
}

std::string clear_magnetic_cache() {
    try {
        OpenMagnetics::magneticsCache.clear();
        return std::to_string(OpenMagnetics::magneticsCache.size());
    }
    catch (const std::exception &exc) {
        return std::string{exc.what()};
    }
}


std::string load_magnetic(std::string key, std::string magneticString, bool expand) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        if (expand) {
            magnetic = OpenMagnetics::magnetic_autocomplete(magnetic);
        }
        OpenMagnetics::magneticsCache.load(std::move(key), std::move(magnetic));

        return std::to_string(OpenMagnetics::magneticsCache.size());
    }
    catch (const std::exception &exc) {
        return std::string{exc.what()};
    }
}

std::string load_magnetics(std::string keysString, std::string magneticsString, bool expand) {
    try {
        json keys = json::parse(keysString);
        json magneticJsons = json::parse(magneticsString);
        for (size_t magneticIndex = 0; magneticIndex < magneticJsons.size(); magneticIndex++) {
            OpenMagnetics::Magnetic magnetic(magneticJsons[magneticIndex]);
            if (expand) {
                magnetic = OpenMagnetics::magnetic_autocomplete(magnetic);
            }
            std::string key = keys[magneticIndex];
            OpenMagnetics::magneticsCache.load(std::move(key), std::move(magnetic));
        }
        return std::to_string(OpenMagnetics::magneticsCache.size());
    }
    catch (const std::exception &exc) {
        return std::string{exc.what()};
    }
}

std::string load_magnetics_from_file(std::string path, bool expand) {
    try {
        std::ifstream in(path);
        if (in) {
            std::string line;
            while (getline(in, line)) {
                json jf = json::parse(line);
                OpenMagnetics::Magnetic magnetic(jf);
                if (expand) {
                    magnetic = OpenMagnetics::magnetic_autocomplete(magnetic);
                }
                std::string key = magnetic.get_manufacturer_info()->get_reference().value();
                OpenMagnetics::magneticsCache.load(std::move(key), std::move(magnetic));
            }
        }
        return std::to_string(OpenMagnetics::magneticsCache.size());
    }
    catch (const std::exception &exc) {
        return std::string{exc.what()};
    }
}

std::string load_magnetics_from_string(std::string database, bool expand) {
    try {
        // Linear-time NDJSON scan. The previous loop did
        //   database.erase(0, pos + 1)
        // per line, which is O(N²) memmove for multi-MB catalogs. We now
        // walk the buffer once with a sliding `start` index and parse
        // each line directly from the original buffer via iterator pair,
        // mirroring MKF's parse_ndjson refactor.
        const size_t n = database.size();
        size_t start = 0;
        while (start < n) {
            size_t pos = database.find('\n', start);
            size_t end = (pos == std::string::npos) ? n : pos;
            if (end > start) {
                json jf = json::parse(database.begin() + start, database.begin() + end);
                OpenMagnetics::Magnetic magnetic(jf);
                if (expand) {
                    magnetic = OpenMagnetics::magnetic_autocomplete(magnetic);
                }
                std::string key = jf["manufacturerInfo"]["reference"];
                OpenMagnetics::magneticsCache.load(std::move(key), std::move(magnetic));
            }
            if (pos == std::string::npos) break;
            start = pos + 1;
        }
        return std::to_string(OpenMagnetics::magneticsCache.size());
    }
    catch (const std::exception &exc) {
        return std::string{exc.what()};
    }
}

// Forward declarations for new topology functions

// SPICE Code Generation forward declarations
EMSCRIPTEN_KEEPALIVE std::string simulate_cmc_lisn_waveforms(std::string cmcInputsString, double inductance);

// ---- LibraryContext + AdviserConstraints wrappers ------------------------
// JS-friendly wrappers: JS passes JSON strings, we parse into structs and
// run the existing advisers with the per-call context.

static OpenMagnetics::AdviserConstraints parse_constraints_json(const std::string& s) {
    OpenMagnetics::AdviserConstraints c;
    if (s.empty()) return c;
    json j = json::parse(s);
    auto fill = [](OpenMagnetics::TypeFilterSet& f, const json& node) {
        if (node.contains("allowed")) for (auto& v : node["allowed"]) f.allowed.insert(v.get<std::string>());
        if (node.contains("blocked")) for (auto& v : node["blocked"]) f.blocked.insert(v.get<std::string>());
    };
    if (j.contains("shapeFamily"))       fill(c.shapeFamily,        j["shapeFamily"]);
    if (j.contains("coreMaterialType"))  fill(c.coreMaterialType,   j["coreMaterialType"]);
    if (j.contains("wireType"))          fill(c.wireType,           j["wireType"]);
    return c;
}

// Persistent LibraryContext owned by the WASM module. JS calls
// library_context_load(jsonString, "merge"|"replace") to populate it, then
// any advised_* function with `useContext=true` will use it.
static OpenMagnetics::LibraryContext g_libraryContext;

void library_context_load(std::string jsonText, std::string modeString) {
    auto mode = (modeString == "replace")
        ? OpenMagnetics::LibraryContext::LoadMode::Replace
        : OpenMagnetics::LibraryContext::LoadMode::Merge;
    g_libraryContext.loadFromString(jsonText, mode);
}

void library_context_clear() {
    g_libraryContext.clear();
}

bool library_context_empty() {
    return g_libraryContext.empty();
}

std::string calculate_advised_cores_with_context(std::string inputsString,
                                                  std::string weightsString,
                                                  int maximumNumberResults,
                                                  std::string coreModeString,
                                                  std::string constraintsString,
                                                  bool useContext) {
    auto constraints = parse_constraints_json(constraintsString);
    const OpenMagnetics::LibraryContext* ctx = useContext ? &g_libraryContext : nullptr;
    return calculate_advised_cores_impl(inputsString, weightsString, maximumNumberResults, coreModeString,
                                        ctx, constraints);
}

std::string calculate_advised_magnetics_with_context(std::string inputsString,
                                                      std::string weightsString,
                                                      int maximumNumberResults,
                                                      std::string coreModeString,
                                                      std::string constraintsString,
                                                      bool useContext) {
    auto constraints = parse_constraints_json(constraintsString);
    const OpenMagnetics::LibraryContext* ctx = useContext ? &g_libraryContext : nullptr;
    return calculate_advised_magnetics_impl(inputsString, weightsString, maximumNumberResults, coreModeString,
                                            ctx, constraints);
}

std::string calculate_advised_wires_with_context(std::string windingString,
                                                 std::string sectionString,
                                                 std::string currentString,
                                                 std::string solidInsulationRequirementsString,
                                                 double temperature,
                                                 uint8_t numberSections,
                                                 size_t maximumNumberResults,
                                                 std::string constraintsString,
                                                 bool useContext) {
    try {
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Winding winding(json::parse(windingString));
        OpenMagnetics::WireSolidInsulationRequirements wireSolidInsulationRequirements(json::parse(solidInsulationRequirementsString));
        Section section(json::parse(sectionString));
        SignalDescriptor current(json::parse(currentString));

        OpenMagnetics::WireAdviser wireAdviser;
        wireAdviser.set_wire_solid_insulation_requirements(wireSolidInsulationRequirements);

        auto constraints = parse_constraints_json(constraintsString);
        const OpenMagnetics::LibraryContext* ctx = useContext ? &g_libraryContext : nullptr;
        auto windingsWithScoring = wireAdviser.get_advised_wire(winding, section, current, temperature,
                                                                numberSections, maximumNumberResults,
                                                                ctx, constraints);

        json results = json();
        results["data"] = json::array();
        for (auto& [advisedWinding, scoring] : windingsWithScoring) {
            json result;
            json windingJson;
            to_json(windingJson, advisedWinding);
            result["winding"] = windingJson;
            result["scoring"] = scoring;
            results["data"].push_back(result);
        }

        return results.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// _with_context twin of calculate_advised_coil (the MagneticBuilder wire
// 'Advise' buttons): useContext=true RAII-swaps the shared catalogs to the
// loaded LibraryContext for the duration of the call, so the CoilAdviser's
// wire pool comes from the inventory instead of the public catalog.
// useContext=false is byte-identical to the classic entry point.
std::string calculate_advised_coil_with_context(std::string masString, bool useContext) {
    try {
        if (!useContext) {
            return calculate_advised_coil(masString);
        }
        auto scope = g_libraryContext.applyScoped();
        return calculate_advised_coil(masString);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// WindingCombination JSON (de)serialization — kept HERE (the consumer), not in MKF's header, which stays
// json-free. Placed in namespace OpenMagnetics so ADL resolves it for the recursive windingCombinations
// vector and for the explicit calls below.
namespace OpenMagnetics {
inline void from_json(const json& j, MagneticCombinator::WindingCombination& x) {
    x.set_connections(j.at("connections").get<std::vector<MAS::ConnectionElement>>());
    x.set_number_turns(j.at("numberTurns").get<size_t>());
    x.set_number_parallels(j.at("numberParallels").get<size_t>());
    x.set_winding_indexes(j.at("windingIndexes").get<std::vector<size_t>>());
    x.set_is_parallel(j.at("isParallel").get<bool>());
    x.set_isolation_side(j.at("isolationSide").get<MAS::IsolationSide>());
    x.set_winding_combinations(j.at("windingCombinations").get<std::vector<MagneticCombinator::WindingCombination>>());
}
inline void to_json(json& j, const MagneticCombinator::WindingCombination& x) {
    j = json::object();
    j["connections"] = x.get_connections();
    j["numberTurns"] = x.get_number_turns();
    j["numberParallels"] = x.get_number_parallels();
    j["windingIndexes"] = x.get_winding_indexes();
    j["isParallel"] = x.get_is_parallel();
    j["isolationSide"] = x.get_isolation_side();
    j["windingCombinations"] = x.get_winding_combinations();
}
}  // namespace OpenMagnetics

// ── Virtual magnetics ────────────────────────────────────────────────────────────────────────────
// Enumerate winding series/parallel recombinations of a real multi-winding magnetic to synthesize
// "virtual" magnetics whose turns ratios + magnetizing inductance match the design Inputs. Returns
// {usedCombinations: {name -> WindingCombination}, virtualMagnetics: [Magnetic]}. Feed a chosen virtual
// magnetic (+ its usedCombinations) back to devirtualize_mas to recover the real part + pin connections.
std::string calculate_virtual_magnetics(std::string magneticString, std::string inputsString) {
    try {
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OpenMagnetics::Inputs inputs(json::parse(inputsString));
        OpenMagnetics::MagneticCombinator combinator;
        auto [usedCombinations, virtualMagnetics] = combinator.calculate_virtual_magnetics(magnetic, inputs);
        json result;
        result["usedCombinations"] = json::object();
        result["virtualMagnetics"] = json::array();
        for (auto& [name, combination] : usedCombinations) {
            json aux; OpenMagnetics::to_json(aux, combination); result["usedCombinations"][name] = aux;
        }
        for (auto& virtualMagnetic : virtualMagnetics) {
            json aux; to_json(aux, virtualMagnetic); result["virtualMagnetics"].push_back(aux);
        }
        return result.dump();
    } catch (const std::exception& e) {
        return std::string("Exception: ") + e.what();
    }
}

// Map a chosen virtual magnetic back to the ORIGINAL real part + the concrete winding connections.
// devirtualizationData is the `usedCombinations` object returned by calculate_virtual_magnetics.
std::string devirtualize_mas(std::string originalMagneticString, std::string virtualMasString,
                             std::string devirtualizationDataString) {
    try {
        OpenMagnetics::Magnetic originalMagnetic(json::parse(originalMagneticString));
        OpenMagnetics::Mas virtualMas(json::parse(virtualMasString));
        std::map<std::string, OpenMagnetics::MagneticCombinator::WindingCombination> devirtualizationData;
        for (auto& [name, val] : json::parse(devirtualizationDataString).items()) {
            OpenMagnetics::MagneticCombinator::WindingCombination wc;
            OpenMagnetics::from_json(val, wc);
            devirtualizationData[name] = wc;
        }
        auto mas = OpenMagnetics::MagneticCombinator::devirtualize_mas(originalMagnetic, virtualMas, devirtualizationData);
        json result; to_json(result, mas);
        return result.dump();
    } catch (const std::exception& e) {
        return std::string("Exception: ") + e.what();
    }
}

EMSCRIPTEN_BINDINGS(my_bindings) {
    function("get_constants", &get_constants);
    function("calculate_virtual_magnetics", &calculate_virtual_magnetics);
    function("devirtualize_mas", &devirtualize_mas);
    function("get_defaults", &get_defaults);
    function("standardize_signal_descriptor", &standardize_signal_descriptor);
    function("calculate_harmonics", &calculate_harmonics);
    function("get_main_harmonic_indexes", &get_main_harmonic_indexes);
    function("get_excitation_harmonic_indexes", &get_excitation_harmonic_indexes);
    function("calculate_processed", &calculate_processed);
    function("calculate_core_data", &calculate_core_data);
    function("calculate_bobbin_data", &calculate_bobbin_data);
    function("get_wire_data", &get_wire_data);
    function("get_wire_data_by_name", &get_wire_data_by_name);
    function("get_wire_data_by_standard_name", &get_wire_data_by_standard_name);
    function("get_strand_by_standard_name", &get_strand_by_standard_name);
    function("get_wire_outer_width_rectangular", &get_wire_outer_width_rectangular);
    function("get_wire_outer_height_rectangular", &get_wire_outer_height_rectangular);
    function("get_wire_outer_diameter_bare_litz", &get_wire_outer_diameter_bare_litz);
    function("get_wire_outer_diameter_served_litz", &get_wire_outer_diameter_served_litz);
    function("get_wire_outer_diameter_insulated_litz", &get_wire_outer_diameter_insulated_litz);
    function("get_wire_outer_diameter_enamelled_round", &get_wire_outer_diameter_enamelled_round);
    function("get_wire_outer_diameter_insulated_round", &get_wire_outer_diameter_insulated_round);
    function("get_wire_conducting_diameter_by_standard_name", &get_wire_conducting_diameter_by_standard_name);
    function("get_outer_dimensions", &get_outer_dimensions);
    function("get_equivalent_wire", &get_equivalent_wire);
    function("get_coating_label", &get_coating_label);
    function("get_wire_coating_by_label", &get_wire_coating_by_label);
    function("get_coating_labels_by_type", &get_coating_labels_by_type);
    function("load_core_data", &load_core_data);
    function("get_material_data", &get_material_data);
    function("get_core_temperature_dependant_parameters", &get_core_temperature_dependant_parameters);
    function("calculate_core_data_from_shape", &calculate_core_data_from_shape);
    function("calculate_all_core_data_from_shapes", &calculate_all_core_data_from_shapes);
    function("get_shape_data", &get_shape_data);
    function("get_available_core_materials", &get_available_core_materials);
    function("get_available_core_materials_with_loss_model", &get_available_core_materials_with_loss_model);
    function("get_core_materials_summary", &get_core_materials_summary);
    function("get_wires_summary", &get_wires_summary);
    function("get_available_core_manufacturers", &get_available_core_manufacturers);
    function("get_available_core_shape_families", &get_available_core_shape_families);
    function("get_supported_core_shape_families", &get_supported_core_shape_families);
    function("get_available_core_shapes", &get_available_core_shapes);
    function("get_available_core_shapes_by_manufacturer", &get_available_core_shapes_by_manufacturer);
    function("get_available_core_shapes_by_family", &get_available_core_shapes_by_family);
    function("get_shape_family_dimensions", &get_shape_family_dimensions);
    function("get_shape_family_subtypes", &get_shape_family_subtypes);
    function("get_available_wires", &get_available_wires);
    function("get_unique_wire_diameters", &get_unique_wire_diameters);
    function("get_planar_thicknesses", &get_planar_thicknesses);
    function("get_planar_wire_by_standard_name", &get_planar_wire_by_standard_name);
    function("get_available_wire_types", &get_available_wire_types);
    function("get_available_wire_standards", &get_available_wire_standards);
    function("calculate_gap_reluctance", &calculate_gap_reluctance);
    function("get_gap_reluctance_model_information", &get_gap_reluctance_model_information);
    function("calculate_inductance_from_number_turns_and_gapping", &calculate_inductance_from_number_turns_and_gapping);
    function("calculate_number_turns_from_gapping_and_inductance", &calculate_number_turns_from_gapping_and_inductance);
    function("calculate_number_turns_from_gapping_and_inductance_legacy", &calculate_number_turns_from_gapping_and_inductance_legacy);
    function("calculate_gapping_from_number_turns_and_inductance", &calculate_gapping_from_number_turns_and_inductance);
    function("calculate_core_losses", &calculate_core_losses);
    function("get_core_losses_model_information", &get_core_losses_model_information);
    function("get_core_temperature_model_information", &get_core_temperature_model_information);
    function("resolve_dimension_with_tolerance", &resolve_dimension_with_tolerance);
    function("calculate_induced_voltage", &calculate_induced_voltage);
    function("calculate_induced_current", &calculate_induced_current);
    function("calculate_reflected_secondary", &calculate_reflected_secondary);
    function("calculate_reflected_primary", &calculate_reflected_primary);
    function("calculate_instantaneous_power", &calculate_instantaneous_power);
    function("calculate_rms_power", &calculate_rms_power);
    function("calculate_basic_processed_data", &calculate_basic_processed_data);
    function("create_waveform", &create_waveform);
    function("scale_waveform_time_to_frequency", &scale_waveform_time_to_frequency);
    function("scale_excitation_time_to_frequency", &scale_excitation_time_to_frequency);
    function("calculate_insulation", &calculate_insulation);
    function("extract_operating_point", &extract_operating_point);
    function("extract_map_column_names", &extract_map_column_names);
    function("extract_column_names", &extract_column_names);
    function("calculate_number_turns", &calculate_number_turns);
    function("calculate_dc_resistance_per_meter", &calculate_dc_resistance_per_meter);
    function("calculate_dc_resistance_per_winding", &calculate_dc_resistance_per_winding);
    function("calculate_dc_losses_per_meter", &calculate_dc_losses_per_meter);
    function("calculate_skin_ac_losses_per_meter", &calculate_skin_ac_losses_per_meter);
    function("calculate_skin_ac_factor", &calculate_skin_ac_factor);
    function("calculate_skin_ac_resistance_per_meter", &calculate_skin_ac_resistance_per_meter);
    function("calculate_effective_current_density", &calculate_effective_current_density);
    function("calculate_effective_skin_depth", &calculate_effective_skin_depth);
    function("get_available_winding_orientations", &get_available_winding_orientations);
    function("get_available_coil_alignments", &get_available_coil_alignments);
    function("check_requirement", &check_requirement);
    function("wind", &wind);
    function("wind_with_columns", &wind_with_columns);
    function("calculate_proportion_per_winding_based_on_wires", &calculate_proportion_per_winding_based_on_wires);
    function("wind_planar", &wind_planar);
    function("wind_by_sections", &wind_by_sections);
    function("wind_by_layers", &wind_by_layers);
    function("wind_by_turns", &wind_by_turns);
    function("delimit_and_compact", &delimit_and_compact);
    function("delimit_and_compact_with_columns", &delimit_and_compact_with_columns);
    function("wind_layers_and_turns_with_columns", &wind_layers_and_turns_with_columns);
    function("get_layers_by_winding_index", &get_layers_by_winding_index);
    function("get_layers_by_section", &get_layers_by_section);
    function("get_sections_description_conduction", &get_sections_description_conduction);
    function("simulate", &simulate);
    function("are_sections_and_layers_fitting", &are_sections_and_layers_fitting);
    function("add_margin_to_section_by_index", &add_margin_to_section_by_index);
    function("check_if_fits", &check_if_fits);
    function("export_magnetic_as_subcircuit", &export_magnetic_as_subcircuit);
    function("export_magnetic_as_symbol", &export_magnetic_as_symbol);
    function("calculate_ac_resistance_coefficients_per_winding", &calculate_ac_resistance_coefficients_per_winding);
    function("sweep_impedance_over_frequency", &sweep_impedance_over_frequency);
    function("sweep_q_factor_over_frequency", &sweep_q_factor_over_frequency);
    function("sweep_winding_resistance_over_frequency", &sweep_winding_resistance_over_frequency);
    function("sweep_resistance_over_frequency", &sweep_resistance_over_frequency);
    function("sweep_core_losses_over_frequency", &sweep_core_losses_over_frequency);
    function("sweep_winding_losses_over_frequency", &sweep_winding_losses_over_frequency);
    function("sweep_magnetizing_inductance_over_frequency", &sweep_magnetizing_inductance_over_frequency);
    function("sweep_magnetizing_inductance_over_temperature", &sweep_magnetizing_inductance_over_temperature);
    function("sweep_magnetizing_inductance_over_dc_bias", &sweep_magnetizing_inductance_over_dc_bias);
    function("load_core_materials", &load_core_materials);
    function("load_core_shapes", &load_core_shapes);
    function("load_wires", &load_wires);
    function("clear_databases", &clear_databases);
    function("library_context_load", &library_context_load);
    function("library_context_clear", &library_context_clear);
    function("library_context_empty", &library_context_empty);
    function("calculate_advised_cores_with_context", &calculate_advised_cores_with_context);
    function("calculate_advised_magnetics_with_context", &calculate_advised_magnetics_with_context);
    function("calculate_advised_wires_with_context", &calculate_advised_wires_with_context);
    function("calculate_advised_coil_with_context", &calculate_advised_coil_with_context);
    function("is_core_material_database_empty", &is_core_material_database_empty);
    function("is_core_shape_database_empty", &is_core_shape_database_empty);
    function("is_wire_database_empty", &is_wire_database_empty);
    function("get_maximum_dimensions", &get_maximum_dimensions);
    function("calculate_advised_cores", &calculate_advised_cores);
    function("calculate_advised_sections", &calculate_advised_sections);
    function("calculate_advised_coil", &calculate_advised_coil);
    function("calculate_advised_wires", &calculate_advised_wires);
    function("get_solid_insulation_requirements_for_wires", &get_solid_insulation_requirements_for_wires);
    function("calculate_advised_magnetics", &calculate_advised_magnetics);
    function("calculate_advised_magnetics_from_catalog", &calculate_advised_magnetics_from_catalog);
    function("calculate_advised_magnetics_from_cache", &calculate_advised_magnetics_from_cache);
    function("get_available_core_filters", &get_available_core_filters);
    function("load_cores", &load_cores);
    function("clear_loaded_cores", &clear_loaded_cores);
    function("calculate_leakage_inductance", &calculate_leakage_inductance);
    function("calculate_inductance_matrix", &calculate_inductance_matrix);
    function("calculate_coupling_coefficient_matrix", &calculate_coupling_coefficient_matrix);
    function("calculate_leakage_inductance_matrix", &calculate_leakage_inductance_matrix);
    function("calculate_stray_capacitance", &calculate_stray_capacitance);
    function("calculate_capacitance_matrix", &calculate_capacitance_matrix);
    function("calculate_maxwell_capacitance_matrix", &calculate_maxwell_capacitance_matrix);
    function("get_available_core_losses_methods", &get_available_core_losses_methods);
    function("get_all_magnetic_field_strength_models", &get_all_magnetic_field_strength_models);
    function("get_all_fringing_effect_models", &get_all_fringing_effect_models);
    function("get_all_reluctance_models", &get_all_reluctance_models);
    function("get_all_winding_skin_effect_models", &get_all_winding_skin_effect_models);
    function("get_all_winding_proximity_effect_models", &get_all_winding_proximity_effect_models);
    function("get_all_stray_capacitance_models", &get_all_stray_capacitance_models);
    function("calculate_resistance_matrix", &calculate_resistance_matrix);
    // Forward converter functions
    
    // SPICE Code Generation functions

    function("get_only_temperature_dependent_indexes", &get_only_temperature_dependent_indexes);
    function("get_only_frequency_dependent_indexes", &get_only_frequency_dependent_indexes);
    function("get_only_magnetic_field_dc_bias_dependent_indexes", &get_only_magnetic_field_dc_bias_dependent_indexes);
    function("create_simple_bobbin_from_core", &create_simple_bobbin_from_core);
    function("create_simple_bobbin_from_core_with_custom_thickness", &create_simple_bobbin_from_core_with_custom_thickness);
    function("create_simple_bobbin_from_core_with_custom_thicknesses", &create_simple_bobbin_from_core_with_custom_thicknesses);
    function("mas_autocomplete", &mas_autocomplete);
    function("magnetic_autocomplete", &magnetic_autocomplete);
    function("calculate_steinmetz_coefficients", &calculate_steinmetz_coefficients);
    function("sweep_volumetric_losses_over_frequency", &sweep_volumetric_losses_over_frequency);
    function("get_initial_permeability_equations", &get_initial_permeability_equations);
    function("get_core_volumetric_losses_equations", &get_core_volumetric_losses_equations);
    function("calculate_complex_permeability", &calculate_complex_permeability);
    function("plot_core", &plot_core);
    function("plot_sections", &plot_sections);
    function("plot_layers", &plot_layers);
    function("plot_turns", &plot_turns);
    function("plot_magnetic", &plot_magnetic);
    function("get_connection_layout", &get_connection_layout);   // ABT #849: connections as data
    function("read_log", &read_log);
    function("plot_magnetic_field", &plot_magnetic_field);
    function("plot_electric_field", &plot_electric_field);
    function("plot_temperature_field", &plot_temperature_field);
    function("plot_wire_losses", &plot_wire_losses);
    function("plot_wire", &plot_wire);
    function("set_interlayer_insulation", &set_interlayer_insulation);
    function("set_intersection_insulation", &set_intersection_insulation);
    function("calculate_filling_factor", &calculate_filling_factor);
    function("get_settings", &get_settings);
    function("set_settings", &set_settings);
    function("reset_settings", &reset_settings);
    function("clear_magnetic_cache", &clear_magnetic_cache);
    function("load_magnetic", &load_magnetic);
    function("load_magnetics", &load_magnetics);
    function("load_magnetics_from_file", &load_magnetics_from_file);
    function("load_magnetics_from_string", &load_magnetics_from_string);
    
    // New topology wizard functions
    
    // New integrated converter processing functions
    
    register_map<std::string, double>("map<string, double>");
    register_map<std::string, std::string>("map<string, string>");
    // register_map<std::string, std::map<std::string, std::string>>("map<string, map<string, string>>");
    register_vector<std::string>("vector<std::string>");
    register_vector<int>("vector<int>");
    register_vector<double>("vector<double>");
    register_vector<size_t>("vector<size_t>");

};

// New topology functions - DAB, LLC, CLLC, PSFB





// ============================================================================
// PSFB / PSHB / AHB - DAB-shaped wizard bindings
// Each pair (calculate_*_inputs / simulate_*_ideal_waveforms) follows the same
// pattern as calculate_dab_inputs / simulate_dab_ideal_waveforms above:
//   - calculate_*: AdvancedX(json).process() -> Inputs -> JSON
//   - simulate_*:  same, then run NgspiceRunner via simulate_and_extract_*
// Returns "Exception: ..." string on failure (calculate path) or
// {"error": "..."} JSON (simulate path) to match DAB / wizard expectations.
// ============================================================================







// SPICE/TDA-based CLLC ideal waveform simulator. Mirrors
// simulate_llc_ideal_waveforms (7817): runs both topology-waveform and
// per-OP extractions through ngspice, returns inputs + converterWaveforms
// + cllcDiagnostics (richer than LLC — includes ZVS margins, resonant
// transition time, peak primary current, peak resonant-cap voltage, and
// the per-segment sub-state sequence).


// New integrated functions for topology processing and magnetic advising



// =========================================================================
// New converter wrappers (Clllc, Cuk, FourSwitchBuckBoost, Weinberg, Zeta)
// =========================================================================
// Pattern mirrors calculate_isolated_buck_inputs / simulate_isolated_buck_ideal_waveforms.
// All five C++ models expose:
//   - Inputs process()  (analytical: builds designRequirements + operating points)
//   - simulate_and_extract_operating_points(...)
//   - simulate_and_extract_topology_waveforms(..., numberOfPeriods)
// Simulate signatures vary per topology (see audit notes inline).

// ---- Clllc (resonant, vector<turnsRatios> + magnetizingInductance) ----



// ---- Cuk (single inductanceL1, no turns ratios in simulate call) ----



// ---- FourSwitchBuckBoost (single inductance, no turns ratios) ----



// ---- Weinberg (scalar turnsRatio + magnetizingInductance) ----



// ---- Zeta (single inductanceL1, no turns ratios) ----



// ── SRC ─────────────────────────────────────────────────────────────────────



// ── Vienna ───────────────────────────────────────────────────────────────────



// ── CurrentTransformer ───────────────────────────────────────────────────────


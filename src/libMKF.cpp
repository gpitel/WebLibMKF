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
#include "converter_models/CommonModeChoke.h"
#include "converter_models/Flyback.h"
#include "converter_models/IsolatedBuck.h"
#include "converter_models/IsolatedBuckBoost.h"
#include "converter_models/Buck.h"
#include "converter_models/Boost.h"
#include "converter_models/Sepic.h"
#include "converter_models/PushPull.h"
#include "converter_models/SingleSwitchForward.h"
#include "converter_models/PowerFactorCorrection.h"
#include "processors/NgspiceRunner.h"
#include "converter_models/ActiveClampForward.h"
#include "converter_models/TwoSwitchForward.h"
#include "converter_models/CommonModeChoke.h"
#include "converter_models/DifferentialModeChoke.h"
#include "converter_models/Llc.h"
#include "converter_models/Cllc.h"
#include "converter_models/Clllc.h"
#include "converter_models/Cuk.h"
#include "converter_models/FourSwitchBuckBoost.h"
#include "converter_models/Weinberg.h"
#include "converter_models/Zeta.h"
#include "converter_models/Dab.h"
#include "converter_models/PhaseShiftedFullBridge.h"
#include "converter_models/PhaseShiftedHalfBridge.h"
#include "converter_models/AsymmetricHalfBridge.h"
#include "converter_models/Src.h"
#include "converter_models/Vienna.h"
#include "converter_models/CurrentTransformer.h"
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
std::string process_converter(std::string topologyName, std::string converterJson, bool useNgspice);
std::string design_magnetics_from_converter(std::string topologyName, std::string converterJson, 
                                             int maxResults, std::string coreModeString, 
                                             bool useNgspice, std::string weightsString);

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
        if (models.find("reluctance") != models.end()) {
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
        if (models.find("reluctance") != models.end()) {
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
        if (models.find("reluctance") != models.end()) {
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
        if (models.find("reluctance") != models.end()) {
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
        if (models.find("reluctance") != models.end()) {
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

    if (!excitation.get_current().value().get_processed().value().get_rms().value()) {
        auto current = excitation.get_current().value();
        auto processed = OpenMagnetics::Inputs::calculate_processed_data(current.get_harmonics().value(), current.get_waveform().value(), true);
        current.set_processed(processed);
        excitation.set_current(current);
    }
    if (!excitation.get_voltage().value().get_processed().value().get_rms().value()) {
        auto voltage = excitation.get_voltage().value();
        auto processed = OpenMagnetics::Inputs::calculate_processed_data(voltage.get_harmonics().value(), voltage.get_waveform().value(), true);
        voltage.set_processed(processed);
        excitation.set_voltage(voltage);
    }

    auto instantaneousPower = OpenMagnetics::Inputs::calculate_instantaneous_power(excitation);

    return instantaneousPower;
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

std::string wind(std::string coilString, size_t repetitions, std::string proportionPerWindingString, std::string patternString, std::string marginPairsString) {
    try {
        auto coilJson = json::parse(coilString);
        auto marginPairs = std::vector<std::vector<double>>(json::parse(marginPairsString));
        
        OpenMagnetics::Settings::GetInstance().set_coil_wind_even_if_not_fit(true);
        OpenMagnetics::Settings::GetInstance().set_coil_delimit_and_compact(true);
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(true);
        
        std::vector<double> proportionPerWinding = json::parse(proportionPerWindingString);
        std::vector<size_t> pattern = json::parse(patternString);
        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        OpenMagnetics::Coil coil;
        coil.set_bobbin(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.preload_margins(marginPairs);

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
            throw std::runtime_error("Turns not created");
        }

        // Explicitly call delimit_and_compact to ensure toroidal additional turns are compacted
        coil.delimit_and_compact();

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
            throw std::runtime_error("Turns not created");
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

        coil.set_bobbin(coilJson["bobbin"]);
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

        coil.set_bobbin(coilJson["bobbin"]);
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

        coil.set_bobbin(coilJson["bobbin"]);
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

std::string delimit_and_compact(std::string coilString) {
    try {
        auto coilJson = json::parse(coilString);

        auto winding = std::vector<OpenMagnetics::Winding>(coilJson["functionalDescription"]);
        auto coilSectionsDescription = std::vector<Section>(coilJson["sectionsDescription"]);
        auto coilLayersDescription = std::vector<Layer>(coilJson["layersDescription"]);
        auto coilTurnsDescription = std::vector<Turn>(coilJson["turnsDescription"]);
        OpenMagnetics::Coil coil;

        process_coil_configuration(coil, coilString);

        coil.set_bobbin(coilJson["bobbin"]);
        coil.set_functional_description(winding);
        coil.set_sections_description(coilSectionsDescription);
        coil.set_layers_description(coilLayersDescription);
        coil.set_turns_description(coilTurnsDescription);
        
        // Preserve groupsDescription if it exists
        if (coilJson.contains("groupsDescription") && !coilJson["groupsDescription"].is_null()) {
            auto groupsDescription = std::vector<OpenMagnetics::Group>(coilJson["groupsDescription"]);
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
        if (models.find("reluctance") != models.end()) {
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

std::string calculate_advised_cores(std::string inputsString, std::string weightsString, int maximumNumberResults, std::string coreModeString){
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
        auto masMagnetics = coreAdviser.get_advised_core(inputs, weights, maximumNumberResults);
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

std::string calculate_advised_magnetics(std::string inputsString, std::string weightsString, int maximumNumberResults, std::string coreModeString){
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
            masMagnetics = magneticAdviser.get_advised_magnetic(inputs, cmcFilterFlow, maximumNumberResults);
        }
        else {
            masMagnetics = magneticAdviser.get_advised_magnetic(inputs, weights, maximumNumberResults);
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
        if (models.find("reluctance") != models.end()) {
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
        if (models.find("reluctance") != models.end()) {
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
        if (models.find("reluctance") != models.end()) {
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

std::string calculate_stray_capacitance(std::string coilString, std::string operatingPointString, std::string modelsData){
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto strayCapacitanceModelName = OpenMagnetics::StrayCapacitanceModels::ALBACH;
        if (models.find("strayCapacitance") != models.end()) {
            OpenMagnetics::from_json(models["strayCapacitance"], strayCapacitanceModelName);
        }

        OpenMagnetics::StrayCapacitance strayCapacitance(strayCapacitanceModelName);
        auto strayCapacitanceOutput = strayCapacitance.calculate_capacitance(coil, operatingPoint);

        json result;
        to_json(result, strayCapacitanceOutput);
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_capacitance_matrix(std::string coilString, std::string modelsData){
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto strayCapacitanceModelName = OpenMagnetics::StrayCapacitanceModels::ALBACH;
        if (models.find("strayCapacitance") != models.end()) {
            OpenMagnetics::from_json(models["strayCapacitance"], strayCapacitanceModelName);
        }

        OpenMagnetics::StrayCapacitance strayCapacitance(strayCapacitanceModelName);
        auto strayCapacitanceOutput = strayCapacitance.calculate_capacitance(coil);

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

std::string calculate_maxwell_capacitance_matrix(std::string coilString, std::string modelsData){
    try {
        OpenMagnetics::Coil coil(json::parse(coilString), false);
        
        std::map<std::string, std::string> models = json::parse(modelsData).get<std::map<std::string, std::string>>();
        
        auto strayCapacitanceModelName = OpenMagnetics::StrayCapacitanceModels::ALBACH;
        if (models.find("strayCapacitance") != models.end()) {
            OpenMagnetics::from_json(models["strayCapacitance"], strayCapacitanceModelName);
        }

        OpenMagnetics::StrayCapacitance strayCapacitance(strayCapacitanceModelName);
        auto strayCapacitanceOutput = strayCapacitance.calculate_capacitance(coil);

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

std::string calculate_capacitance_models_between_windings(double energy, double voltageDrop, double relativeTurnsRatio){
    try {
        auto result = OpenMagnetics::StrayCapacitance::calculate_capacitance_models_between_windings(energy, voltageDrop, relativeTurnsRatio);
        
        json resultJson;
        
        // Serialize SixCapacitorNetworkPerWinding
        resultJson["sixCapacitorNetwork"]["c1"] = result.first.get_c1();
        resultJson["sixCapacitorNetwork"]["c2"] = result.first.get_c2();
        resultJson["sixCapacitorNetwork"]["c3"] = result.first.get_c3();
        resultJson["sixCapacitorNetwork"]["c4"] = result.first.get_c4();
        resultJson["sixCapacitorNetwork"]["c5"] = result.first.get_c5();
        resultJson["sixCapacitorNetwork"]["c6"] = result.first.get_c6();
        
        // Serialize TripoleCapacitancePerWinding
        resultJson["tripoleCapacitance"]["c1"] = result.second.get_c1();
        resultJson["tripoleCapacitance"]["c2"] = result.second.get_c2();
        resultJson["tripoleCapacitance"]["c3"] = result.second.get_c3();
        
        return resultJson.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

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

std::string calculate_flyback_inputs(std::string flybackInputsString){
    try {
        json flybackInputsJson = json::parse(flybackInputsString);

        OpenMagnetics::Flyback flybackInputs(flybackInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (flybackInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = flybackInputsJson["numberOfPeriods"].get<size_t>();
        }
        flybackInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto inputs = flybackInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            // Flat fields (back-compat): first OP's values. perOp is the
            // authoritative per-operating-point breakdown — wizard table
            // reads this.
            const auto& names = flybackInputs.get_per_op_name();
            const auto& dC   = flybackInputs.get_per_op_duty_cycle();
            const auto& fsw  = flybackInputs.get_per_op_switching_frequency();
            const auto& iAvg = flybackInputs.get_per_op_primary_average_current();
            const auto& iPP  = flybackInputs.get_per_op_primary_peak_to_peak();
            const auto& iPk  = flybackInputs.get_per_op_primary_peak_current();
            const auto& iSec = flybackInputs.get_per_op_secondary_peak_current();
            const auto& ccm  = flybackInputs.get_per_op_is_ccm();
            diag["dutyCycle"]             = dC.empty()   ? flybackInputs.get_last_duty_cycle()              : dC.front();
            diag["switchingFrequency"]    = fsw.empty()  ? flybackInputs.get_last_switching_frequency()     : fsw.front();
            diag["primaryAverageCurrent"] = iAvg.empty() ? flybackInputs.get_last_primary_average_current() : iAvg.front();
            diag["primaryPeakToPeak"]     = iPP.empty()  ? flybackInputs.get_last_primary_peak_to_peak()    : iPP.front();
            diag["primaryPeakCurrent"]    = iPk.empty()  ? flybackInputs.get_last_primary_peak_current()    : iPk.front();
            diag["secondaryPeakCurrent"]  = iSec.empty() ? flybackInputs.get_last_secondary_peak_current()  : iSec.front();
            diag["isCcm"]                 = ccm.empty()  ? flybackInputs.get_last_is_ccm()                  : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["switchingFrequency"]    = fsw[i];
                row["primaryAverageCurrent"] = iAvg[i];
                row["primaryPeakToPeak"]     = iPP[i];
                row["primaryPeakCurrent"]    = iPk[i];
                row["secondaryPeakCurrent"]  = iSec[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["flybackDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_flyback_inputs(std::string flybackInputsString){
    try {
        json flybackInputsJson = json::parse(flybackInputsString);

        OpenMagnetics::AdvancedFlyback flybackInputs(flybackInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (flybackInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = flybackInputsJson["numberOfPeriods"].get<size_t>();
        }
        flybackInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto inputs = flybackInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = flybackInputs.get_per_op_name();
            const auto& dC   = flybackInputs.get_per_op_duty_cycle();
            const auto& fsw  = flybackInputs.get_per_op_switching_frequency();
            const auto& iAvg = flybackInputs.get_per_op_primary_average_current();
            const auto& iPP  = flybackInputs.get_per_op_primary_peak_to_peak();
            const auto& iPk  = flybackInputs.get_per_op_primary_peak_current();
            const auto& iSec = flybackInputs.get_per_op_secondary_peak_current();
            const auto& ccm  = flybackInputs.get_per_op_is_ccm();
            diag["dutyCycle"]             = dC.empty()   ? flybackInputs.get_last_duty_cycle()              : dC.front();
            diag["switchingFrequency"]    = fsw.empty()  ? flybackInputs.get_last_switching_frequency()     : fsw.front();
            diag["primaryAverageCurrent"] = iAvg.empty() ? flybackInputs.get_last_primary_average_current() : iAvg.front();
            diag["primaryPeakToPeak"]     = iPP.empty()  ? flybackInputs.get_last_primary_peak_to_peak()    : iPP.front();
            diag["primaryPeakCurrent"]    = iPk.empty()  ? flybackInputs.get_last_primary_peak_current()    : iPk.front();
            diag["secondaryPeakCurrent"]  = iSec.empty() ? flybackInputs.get_last_secondary_peak_current()  : iSec.front();
            diag["isCcm"]                 = ccm.empty()  ? flybackInputs.get_last_is_ccm()                  : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["switchingFrequency"]    = fsw[i];
                row["primaryAverageCurrent"] = iAvg[i];
                row["primaryPeakToPeak"]     = iPP[i];
                row["primaryPeakCurrent"]    = iPk[i];
                row["secondaryPeakCurrent"]  = iSec[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["flybackDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_flyback_ideal_waveforms(std::string flybackInputsString){
    try {
        json flybackInputsJson = json::parse(flybackInputsString);

        // Detect if this is an AdvancedFlyback (user knows design) or regular Flyback (help with design)
        // AdvancedFlyback has "desiredInductance" field, Flyback has "currentRippleRatio" field
        bool isAdvancedFlyback = flybackInputsJson.contains("desiredInductance");
        
        DesignRequirements designRequirements;
        std::vector<double> turnsRatios;
        double magnetizingInductance;
        
        // Create a unique_ptr to hold either Flyback or AdvancedFlyback
        // AdvancedFlyback inherits from Flyback so we can use the base pointer
        std::unique_ptr<OpenMagnetics::Flyback> flybackPtr;
        
        if (isAdvancedFlyback) {
            // User knows the design they want - use AdvancedFlyback
            // AdvancedFlyback has its own process() method that uses desiredInductance/desiredTurnsRatios/desiredDutyCycle
            auto advancedFlybackPtr = std::make_unique<OpenMagnetics::AdvancedFlyback>(flybackInputsJson);
            
            // Get design parameters directly from the AdvancedFlyback object
            magnetizingInductance = advancedFlybackPtr->get_desired_inductance();
            turnsRatios = advancedFlybackPtr->get_desired_turns_ratios();
            
            // Build designRequirements manually from the desired values
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto turnsRatio : turnsRatios) {
                DimensionWithTolerance turnsRatioWithTolerance;
                turnsRatioWithTolerance.set_nominal(turnsRatio);
                designRequirements.get_mutable_turns_ratios().push_back(turnsRatioWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            std::vector<IsolationSide> isolationSides;
            for (size_t windingIndex = 0; windingIndex < turnsRatios.size() + 1; ++windingIndex) {
                isolationSides.push_back(OpenMagnetics::get_isolation_side_from_index(windingIndex));
            }
            designRequirements.set_isolation_sides(isolationSides);
            designRequirements.set_topology(MAS::Topology::FLYBACK_CONVERTER);
            
            // Move the AdvancedFlyback into the base pointer (polymorphism)
            flybackPtr = std::move(advancedFlybackPtr);
        } else {
            // Help with design - use regular Flyback
            flybackPtr = std::make_unique<OpenMagnetics::Flyback>(flybackInputsJson);
            designRequirements = flybackPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
            
            // Extract magnetizing inductance using the same helper as
            // Topology::process() — defaults to NOMINAL, falls back to
            // (min+max)/2 or whichever bound is available.
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate magnetizing inductance");
            }
        }
        
        // Verify ngspice is available - required for simulation
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (flybackInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = flybackInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (flybackInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = flybackInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        flybackPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
        // Use ngspice-based simulation for accurate waveforms
        auto topologyWaveforms = flybackPtr->simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance, numberOfPeriods);
        
        // Also get the operating points for the magnetic data
        auto operatingPoints = flybackPtr->simulate_and_extract_operating_points(turnsRatios, magnetizingInductance, numberOfPeriods);
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Flyback diagnostics — measured from the ngspice trace (NOT analytical
        // predictions). The OperatingPoint returned by simulate_and_extract_*
        // already carries ProcessedWaveform data computed from the simulation samples
        // via CircuitSimulationReader::extract_operating_point →
        // Inputs::calculate_basic_processed_data. We read those processed
        // fields directly. THROW on any missing required signal — Vienna-style
        // "no fallbacks, no defaults": if the simulator didn't produce a
        // primary winding with current+voltage waveforms and processed data,
        // that's a loud bug, not a number to paper over.
        if (operatingPoints.empty()) {
            throw std::runtime_error("Flyback simulation returned no operating points; cannot extract diagnostics");
        }
        {
            // Note: SignalDescriptor::get_waveform/get_processed/get_current/...
            // all return std::optional<T> BY VALUE in MAS — chaining
            // `.get_current()->get_waveform()->get_data()` would dangle.
            // Capture each step in a local to keep the optionals alive.
            const auto& op = operatingPoints[0];
            const auto& excitations = op.get_excitations_per_winding();
            if (excitations.size() < 2) {
                throw std::runtime_error("Flyback simulation: expected at least primary + 1 secondary winding, got " + std::to_string(excitations.size()));
            }
            const auto& primExc = excitations[0];
            const auto& secExc  = excitations[1];

            auto primCurrentOpt = primExc.get_current();
            auto primVoltageOpt = primExc.get_voltage();
            auto secCurrentOpt  = secExc.get_current();
            if (!primCurrentOpt) throw std::runtime_error("Flyback simulation: primary current SignalDescriptor missing");
            if (!primVoltageOpt) throw std::runtime_error("Flyback simulation: primary voltage SignalDescriptor missing");
            if (!secCurrentOpt)  throw std::runtime_error("Flyback simulation: secondary[0] current SignalDescriptor missing");

            auto primCurrentWfOpt = primCurrentOpt->get_waveform();
            auto primVoltageWfOpt = primVoltageOpt->get_waveform();
            if (!primCurrentWfOpt) throw std::runtime_error("Flyback simulation: primary current waveform missing");
            if (!primVoltageWfOpt) throw std::runtime_error("Flyback simulation: primary voltage waveform missing");

            const auto& iPrimData = primCurrentWfOpt->get_data();
            if (iPrimData.empty()) {
                throw std::runtime_error("Flyback simulation: primary current waveform data is empty");
            }

            auto iPrimProcOpt = primCurrentOpt->get_processed();
            auto vPrimProcOpt = primVoltageOpt->get_processed();
            auto iSecProcOpt  = secCurrentOpt->get_processed();
            if (!iPrimProcOpt) throw std::runtime_error("Flyback simulation: processed data missing on primary current");
            if (!vPrimProcOpt) throw std::runtime_error("Flyback simulation: processed data missing on primary voltage");
            if (!iSecProcOpt)  throw std::runtime_error("Flyback simulation: processed data missing on secondary current");

            auto require = [](const std::optional<double>& v, const char* field) {
                if (!v) throw std::runtime_error(std::string("Flyback simulation: missing processed.") + field);
                return v.value();
            };

            // CCM ↔ DCM: in CCM the primary current valley is bounded above
            // zero (DC offset of the ramp); in DCM the current returns to 0
            // during the OFF interval. 1% threshold is a robust no-fallback
            // discriminator that doesn't depend on label classification.
            double iPrimMin = *std::min_element(iPrimData.begin(), iPrimData.end());
            double iPrimMax = *std::max_element(iPrimData.begin(), iPrimData.end());
            if (!(iPrimMax > 0.0)) {
                throw std::runtime_error("Flyback simulation: primary current peak is non-positive (" + std::to_string(iPrimMax) + " A) — cannot derive mode from trace");
            }
            bool isCcm = (iPrimMin > 0.01 * iPrimMax);

            // Duty cycle comes from the actual SPICE PULSE-source ON time
            // (Flyback::generate_ngspice_circuit sets lastDutyCycle from
            // the same `dutyCycle` it feeds to the `Vpwm ... PULSE(...)`
            // line). This is the duty the simulator actually ran with —
            // independent of trace-shape heuristics that misread bipolar
            // primary voltage waveforms as 50%.
            double dutyCycleFromSpiceInputs = flybackPtr->get_last_duty_cycle();
            if (!(dutyCycleFromSpiceInputs > 0.0 && dutyCycleFromSpiceInputs < 1.0)) {
                throw std::runtime_error("Flyback simulation: SPICE-input dutyCycle out of (0,1): " + std::to_string(dutyCycleFromSpiceInputs));
            }

            // Flat fields = trace-measured OP 0 (the one shown in the
            // chart). perOp[] = analytical per-OP table (populated by
            // process_operating_points below).
            json diag;
            diag["dutyCycle"]             = dutyCycleFromSpiceInputs;
            diag["switchingFrequency"]    = primExc.get_frequency();
            diag["primaryAverageCurrent"] = require(iPrimProcOpt->get_average(), "average (primary current)");
            diag["primaryPeakToPeak"]     = require(iPrimProcOpt->get_peak_to_peak(), "peak_to_peak (primary current)");
            diag["primaryPeakCurrent"]    = require(iPrimProcOpt->get_positive_peak(), "positive_peak (primary current)");
            diag["secondaryPeakCurrent"]  = require(iSecProcOpt->get_peak(), "peak (secondary current)");
            diag["isCcm"]                 = isCcm;

            // Populate analytical per-OP vectors. process_operating_points
            // iterates inputVoltages × OPs and pushes a snapshot per call.
            flybackPtr->process_operating_points(turnsRatios, magnetizingInductance);
            const auto& names = flybackPtr->get_per_op_name();
            const auto& dC   = flybackPtr->get_per_op_duty_cycle();
            const auto& fsw  = flybackPtr->get_per_op_switching_frequency();
            const auto& iAvg = flybackPtr->get_per_op_primary_average_current();
            const auto& iPP  = flybackPtr->get_per_op_primary_peak_to_peak();
            const auto& iPk  = flybackPtr->get_per_op_primary_peak_current();
            const auto& iSec = flybackPtr->get_per_op_secondary_peak_current();
            const auto& ccm  = flybackPtr->get_per_op_is_ccm();
            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["switchingFrequency"]    = fsw[i];
                row["primaryAverageCurrent"] = iAvg[i];
                row["primaryPeakToPeak"]     = iPP[i];
                row["primaryPeakCurrent"]    = iPk[i];
                row["secondaryPeakCurrent"]  = iSec[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["flybackDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Note: simulate_flyback_with_magnetic uses ngspice which is not available in WASM.
// This function is kept for PyMKF (native) usage but will throw an error in browser.
std::string simulate_flyback_with_magnetic(std::string flybackInputsString, std::string magneticString){
    try {
        return "Exception: ngspice-based simulation is not available in browser WASM. Use native PyMKF for real magnetic simulation.";
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// SPICE Code Generation Functions - Returns the ngspice netlist for a converter
EMSCRIPTEN_KEEPALIVE std::string generate_flyback_ngspice_circuit(std::string flybackInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json flybackInputsJson = json::parse(flybackInputsString);
        
        bool isAdvancedFlyback = flybackInputsJson.contains("desiredInductance");
        
        std::unique_ptr<OpenMagnetics::Flyback> flybackPtr;
        std::vector<double> turnsRatios;
        double magnetizingInductance;
        
        if (isAdvancedFlyback) {
            auto advancedFlybackPtr = std::make_unique<OpenMagnetics::AdvancedFlyback>(flybackInputsJson);
            magnetizingInductance = advancedFlybackPtr->get_desired_inductance();
            turnsRatios = advancedFlybackPtr->get_desired_turns_ratios();
            flybackPtr = std::move(advancedFlybackPtr);
        } else {
            flybackPtr = std::make_unique<OpenMagnetics::Flyback>(flybackInputsJson);
            auto designRequirements = flybackPtr->process_design_requirements();
            
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate magnetizing inductance");
            }
        }
        
        std::string netlist = flybackPtr->generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Generic template for converters with turnsRatios and magnetizingInductance
template<typename ConverterType, typename AdvancedConverterType>
std::string generate_converter_ngspice_circuit_helper(std::string inputsString, size_t inputVoltageIndex, size_t operatingPointIndex, const std::string& desiredFieldName) {
    try {
        json inputsJson = json::parse(inputsString);
        
        bool isAdvanced = inputsJson.contains(desiredFieldName);
        
        std::vector<double> turnsRatios;
        double magnetizingInductance;
        
        std::string netlist;
        if (isAdvanced) {
            AdvancedConverterType converter(inputsJson);
            magnetizingInductance = converter.get_desired_inductance();
            turnsRatios = converter.get_desired_turns_ratios();
            netlist = converter.generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
        } else {
            ConverterType converter(inputsJson);
            auto designRequirements = converter.process_design_requirements();
            
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate magnetizing inductance");
            }
            
            netlist = converter.generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
        }
        
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Buck SPICE generation (uses inductance directly, not turns ratios)
EMSCRIPTEN_KEEPALIVE std::string generate_buck_ngspice_circuit(std::string buckInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json buckInputsJson = json::parse(buckInputsString);
        
        bool isAdvancedBuck = buckInputsJson.contains("desiredInductance");
        
        double inductance;
        
        std::string netlist;
        if (isAdvancedBuck) {
            OpenMagnetics::AdvancedBuck buck(buckInputsJson);
            inductance = buck.get_desired_inductance();
            netlist = buck.generate_ngspice_circuit(inductance, inputVoltageIndex, operatingPointIndex);
        } else {
            OpenMagnetics::Buck buck(buckInputsJson);
            auto designRequirements = buck.process_design_requirements();
            
            inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
            
            netlist = buck.generate_ngspice_circuit(inductance, inputVoltageIndex, operatingPointIndex);
        }
        
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Boost SPICE generation (uses inductance directly)
EMSCRIPTEN_KEEPALIVE std::string generate_boost_ngspice_circuit(std::string boostInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json boostInputsJson = json::parse(boostInputsString);
        
        bool isAdvancedBoost = boostInputsJson.contains("desiredInductance");
        
        double inductance;
        
        std::string netlist;
        if (isAdvancedBoost) {
            OpenMagnetics::AdvancedBoost boost(boostInputsJson);
            inductance = boost.get_desired_inductance();
            netlist = boost.generate_ngspice_circuit(inductance, inputVoltageIndex, operatingPointIndex);
        } else {
            OpenMagnetics::Boost boost(boostInputsJson);
            auto designRequirements = boost.process_design_requirements();
            
            inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
            
            netlist = boost.generate_ngspice_circuit(inductance, inputVoltageIndex, operatingPointIndex);
        }
        
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// SEPIC SPICE generation (uses inductanceL1 directly)
EMSCRIPTEN_KEEPALIVE std::string generate_sepic_ngspice_circuit(std::string sepicInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json sepicInputsJson = json::parse(sepicInputsString);

        bool isAdvancedSepic = sepicInputsJson.contains("desiredInductance");

        double inductanceL1;

        std::string netlist;
        if (isAdvancedSepic) {
            OpenMagnetics::AdvancedSepic sepic(sepicInputsJson);
            inductanceL1 = sepic.get_desired_inductance();
            netlist = sepic.generate_ngspice_circuit(inductanceL1, inputVoltageIndex, operatingPointIndex);
        } else {
            OpenMagnetics::Sepic sepic(sepicInputsJson);
            auto designRequirements = sepic.process_design_requirements();

            inductanceL1 = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductanceL1 > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }

            netlist = sepic.generate_ngspice_circuit(inductanceL1, inputVoltageIndex, operatingPointIndex);
        }

        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// PushPull SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_push_pull_ngspice_circuit(std::string pushPullInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    return generate_converter_ngspice_circuit_helper<OpenMagnetics::PushPull, OpenMagnetics::AdvancedPushPull>(pushPullInputsString, inputVoltageIndex, operatingPointIndex, "desiredInductance");
}

// Forward SPICE generation (Single Switch)
EMSCRIPTEN_KEEPALIVE std::string generate_forward_ngspice_circuit(std::string forwardInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    return generate_converter_ngspice_circuit_helper<OpenMagnetics::SingleSwitchForward, OpenMagnetics::AdvancedSingleSwitchForward>(forwardInputsString, inputVoltageIndex, operatingPointIndex, "desiredInductance");
}

// Two Switch Forward SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_two_switch_forward_ngspice_circuit(std::string forwardInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    return generate_converter_ngspice_circuit_helper<OpenMagnetics::TwoSwitchForward, OpenMagnetics::AdvancedTwoSwitchForward>(forwardInputsString, inputVoltageIndex, operatingPointIndex, "desiredInductance");
}

// Active Clamp Forward SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_active_clamp_forward_ngspice_circuit(std::string forwardInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    return generate_converter_ngspice_circuit_helper<OpenMagnetics::ActiveClampForward, OpenMagnetics::AdvancedActiveClampForward>(forwardInputsString, inputVoltageIndex, operatingPointIndex, "desiredInductance");
}

// Isolated Buck SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_isolated_buck_ngspice_circuit(std::string isolatedBuckInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    return generate_converter_ngspice_circuit_helper<OpenMagnetics::IsolatedBuck, OpenMagnetics::AdvancedIsolatedBuck>(isolatedBuckInputsString, inputVoltageIndex, operatingPointIndex, "desiredInductance");
}

// Isolated Buck Boost SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_isolated_buck_boost_ngspice_circuit(std::string isolatedBuckBoostInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    return generate_converter_ngspice_circuit_helper<OpenMagnetics::IsolatedBuckBoost, OpenMagnetics::AdvancedIsolatedBuckBoost>(isolatedBuckBoostInputsString, inputVoltageIndex, operatingPointIndex, "desiredInductance");
}

// LLC SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_llc_ngspice_circuit(std::string llcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json llcInputsJson = json::parse(llcInputsString);
        OpenMagnetics::Llc llc(llcInputsJson);
        
        // For LLC, we need to extract from functional description or design requirements
        auto designRequirements = llc.process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            turnsRatios.push_back(tr.get_nominal().value());
        }
        
        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to calculate magnetizing inductance");
        }
        
        std::string netlist = llc.generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// CLLC SPICE generation — uses CllcResonantParameters from
// calculate_resonant_parameters(); turnsRatio is scalar (single secondary).
EMSCRIPTEN_KEEPALIVE std::string generate_cllc_ngspice_circuit(std::string cllcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json cllcInputsJson = json::parse(cllcInputsString);
        OpenMagnetics::CllcConverter cllc(cllcInputsJson);
        auto params = cllc.calculate_resonant_parameters();
        double turnsRatio = params.turnsRatio;
        std::string netlist = cllc.generate_ngspice_circuit(turnsRatio, params, inputVoltageIndex, operatingPointIndex);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Cuk SPICE generation (inductance-shaped, mirrors generate_sepic_ngspice_circuit)
EMSCRIPTEN_KEEPALIVE std::string generate_cuk_ngspice_circuit(std::string cukInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json cukInputsJson = json::parse(cukInputsString);

        if (cukInputsJson.contains("desiredInductance")) {
            OpenMagnetics::AdvancedCuk cuk(cukInputsJson);
            return cuk.generate_ngspice_circuit(cuk.get_desired_inductance(), inputVoltageIndex, operatingPointIndex);
        }
        OpenMagnetics::Cuk cuk(cukInputsJson);
        auto designRequirements = cuk.process_design_requirements();
        double inductanceL1 = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(inductanceL1 > 0)) {
            throw std::runtime_error("Cuk: unable to calculate inductance");
        }
        return cuk.generate_ngspice_circuit(inductanceL1, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Zeta SPICE generation (inductance-shaped)
EMSCRIPTEN_KEEPALIVE std::string generate_zeta_ngspice_circuit(std::string zetaInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json zetaInputsJson = json::parse(zetaInputsString);

        if (zetaInputsJson.contains("desiredInductance")) {
            OpenMagnetics::AdvancedZeta zeta(zetaInputsJson);
            return zeta.generate_ngspice_circuit(zeta.get_desired_inductance(), inputVoltageIndex, operatingPointIndex);
        }
        OpenMagnetics::Zeta zeta(zetaInputsJson);
        auto designRequirements = zeta.process_design_requirements();
        double inductanceL1 = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(inductanceL1 > 0)) {
            throw std::runtime_error("Zeta: unable to calculate inductance");
        }
        return zeta.generate_ngspice_circuit(inductanceL1, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Four-Switch Buck-Boost SPICE generation (inductance-shaped)
EMSCRIPTEN_KEEPALIVE std::string generate_four_switch_buck_boost_ngspice_circuit(std::string fsbbInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json fsbbInputsJson = json::parse(fsbbInputsString);

        if (fsbbInputsJson.contains("desiredInductance")) {
            OpenMagnetics::AdvancedFourSwitchBuckBoost fsbb(fsbbInputsJson);
            return fsbb.generate_ngspice_circuit(fsbb.get_desired_inductance(), inputVoltageIndex, operatingPointIndex);
        }
        OpenMagnetics::FourSwitchBuckBoost fsbb(fsbbInputsJson);
        auto designRequirements = fsbb.process_design_requirements();
        double inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(inductance > 0)) {
            throw std::runtime_error("FourSwitchBuckBoost: unable to calculate inductance");
        }
        return fsbb.generate_ngspice_circuit(inductance, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Weinberg SPICE generation (scalar turnsRatio + magnetizing inductance)
EMSCRIPTEN_KEEPALIVE std::string generate_weinberg_ngspice_circuit(std::string weinbergInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json weinbergInputsJson = json::parse(weinbergInputsString);

        if (weinbergInputsJson.contains("desiredInductance") && weinbergInputsJson.contains("desiredTurnsRatio")) {
            OpenMagnetics::AdvancedWeinberg weinberg(weinbergInputsJson);
            return weinberg.generate_ngspice_circuit(weinberg.get_desired_turns_ratio(), weinberg.get_desired_inductance(), inputVoltageIndex, operatingPointIndex);
        }
        OpenMagnetics::Weinberg weinberg(weinbergInputsJson);
        auto designRequirements = weinberg.process_design_requirements();
        if (designRequirements.get_turns_ratios().empty() || !designRequirements.get_turns_ratios()[0].get_nominal()) {
            throw std::runtime_error("Weinberg: process_design_requirements produced no turns ratio");
        }
        double turnsRatio = designRequirements.get_turns_ratios()[0].get_nominal().value();
        double magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Weinberg: no magnetizing inductance available");
        }
        return weinberg.generate_ngspice_circuit(turnsRatio, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// CLLLC SPICE generation (turnsRatios vector + magnetizing inductance, mirrors SRC)
EMSCRIPTEN_KEEPALIVE std::string generate_clllc_ngspice_circuit(std::string clllcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json clllcInputsJson = json::parse(clllcInputsString);

        bool isAdvanced = clllcInputsJson.contains("desiredMagnetizingInductance");
        std::unique_ptr<OpenMagnetics::Clllc> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedClllc>(clllcInputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Clllc>(clllcInputsJson);
        }

        auto designRequirements = model->process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("CLLLC: process_design_requirements produced no turns ratios");
        }
        double magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("CLLLC: no magnetizing inductance available");
        }
        return model->generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// PSHB SPICE generation (turnsRatios vector + magnetizing inductance, mirrors SRC)
EMSCRIPTEN_KEEPALIVE std::string generate_pshb_ngspice_circuit(std::string pshbInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json pshbInputsJson = json::parse(pshbInputsString);

        bool isAdvanced = pshbInputsJson.contains("desiredMagnetizingInductance");
        std::unique_ptr<OpenMagnetics::Pshb> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedPshb>(pshbInputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Pshb>(pshbInputsJson);
        }

        auto designRequirements = model->process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("PSHB: process_design_requirements produced no turns ratios");
        }
        double magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("PSHB: no magnetizing inductance available");
        }
        return model->generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// AHB SPICE generation (turnsRatios vector + magnetizing inductance, mirrors SRC)
EMSCRIPTEN_KEEPALIVE std::string generate_ahb_ngspice_circuit(std::string ahbInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json ahbInputsJson = json::parse(ahbInputsString);

        bool isAdvanced = ahbInputsJson.contains("desiredMagnetizingInductance");
        std::unique_ptr<OpenMagnetics::AsymmetricHalfBridge> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedAsymmetricHalfBridge>(ahbInputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::AsymmetricHalfBridge>(ahbInputsJson);
        }

        auto designRequirements = model->process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("AHB: process_design_requirements produced no turns ratios");
        }
        double magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("AHB: no magnetizing inductance available");
        }
        return model->generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// SRC SPICE generation — generates the ngspice netlist string (no simulation)
EMSCRIPTEN_KEEPALIVE std::string generate_src_ngspice_circuit(std::string srcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json srcInputsJson = json::parse(srcInputsString);

        bool isAdvanced = srcInputsJson.contains("desiredTurnsRatios") ||
                          srcInputsJson.contains("desiredResonantInductance") ||
                          srcInputsJson.contains("desiredResonantCapacitance");

        std::unique_ptr<OpenMagnetics::Src> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedSrc>(srcInputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Src>(srcInputsJson);
        }

        auto designRequirements = model->process_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("SRC: process_design_requirements produced no turns ratios");
        }

        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("SRC: no magnetizing inductance available");
        }

        return model->generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// DAB SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_dab_ngspice_circuit(std::string dabInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json dabInputsJson = json::parse(dabInputsString);
        OpenMagnetics::Dab dab(dabInputsJson);
        
        auto designRequirements = dab.process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            turnsRatios.push_back(tr.get_nominal().value());
        }
        
        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to calculate magnetizing inductance");
        }
        
        std::string netlist = dab.generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// Phase Shifted Full Bridge SPICE generation
EMSCRIPTEN_KEEPALIVE std::string generate_psfb_ngspice_circuit(std::string psfbInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json psfbInputsJson = json::parse(psfbInputsString);
        OpenMagnetics::Psfb psfb(psfbInputsJson);
        
        auto designRequirements = psfb.process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            turnsRatios.push_back(tr.get_nominal().value());
        }
        
        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to calculate magnetizing inductance");
        }
        
        std::string netlist = psfb.generate_ngspice_circuit(turnsRatios, magnetizingInductance, inputVoltageIndex, operatingPointIndex);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_isolated_buck_inputs(std::string isolatedBuckInputsString){
    try {
        json isolatedBuckInputsJson = json::parse(isolatedBuckInputsString);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (isolatedBuckInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = isolatedBuckInputsJson["numberOfPeriods"].get<size_t>();
        }

        OpenMagnetics::IsolatedBuck isolatedBuckInputs(isolatedBuckInputsJson);
        auto inputs = isolatedBuckInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = isolatedBuckInputs.get_per_op_name();
            const auto& v_duty_cycle = isolatedBuckInputs.get_per_op_duty_cycle();
            const auto& v_magnetizing_current_ripple = isolatedBuckInputs.get_per_op_magnetizing_current_ripple();
            const auto& v_primary_average_current = isolatedBuckInputs.get_per_op_primary_average_current();
            const auto& v_primary_peak_current = isolatedBuckInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = isolatedBuckInputs.get_per_op_secondary_peak_current();
            const auto& v_is_ccm = isolatedBuckInputs.get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? isolatedBuckInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["magnetizingCurrentRipple"] = v_magnetizing_current_ripple.empty() ? isolatedBuckInputs.get_last_magnetizing_current_ripple() : v_magnetizing_current_ripple.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? isolatedBuckInputs.get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? isolatedBuckInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? isolatedBuckInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? isolatedBuckInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["magnetizingCurrentRipple"] = v_magnetizing_current_ripple[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["isolatedBuckDiagnostics"] = diag;
        }

        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_isolated_buck_inputs(std::string isolatedBuckInputsString){
    try {
        json isolatedBuckInputsJson = json::parse(isolatedBuckInputsString);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (isolatedBuckInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = isolatedBuckInputsJson["numberOfPeriods"].get<size_t>();
        }

        OpenMagnetics::AdvancedIsolatedBuck isolatedBuckInputs(isolatedBuckInputsJson);
        auto inputs = isolatedBuckInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = isolatedBuckInputs.get_per_op_name();
            const auto& v_duty_cycle = isolatedBuckInputs.get_per_op_duty_cycle();
            const auto& v_magnetizing_current_ripple = isolatedBuckInputs.get_per_op_magnetizing_current_ripple();
            const auto& v_primary_average_current = isolatedBuckInputs.get_per_op_primary_average_current();
            const auto& v_primary_peak_current = isolatedBuckInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = isolatedBuckInputs.get_per_op_secondary_peak_current();
            const auto& v_is_ccm = isolatedBuckInputs.get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? isolatedBuckInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["magnetizingCurrentRipple"] = v_magnetizing_current_ripple.empty() ? isolatedBuckInputs.get_last_magnetizing_current_ripple() : v_magnetizing_current_ripple.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? isolatedBuckInputs.get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? isolatedBuckInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? isolatedBuckInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? isolatedBuckInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["magnetizingCurrentRipple"] = v_magnetizing_current_ripple[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["isolatedBuckDiagnostics"] = diag;
        }

        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_isolated_buck_boost_inputs(std::string isolatedBuckBoostInputsString){
    try {
        json isolatedBuckBoostInputsJson = json::parse(isolatedBuckBoostInputsString);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (isolatedBuckBoostInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = isolatedBuckBoostInputsJson["numberOfPeriods"].get<size_t>();
        }

        OpenMagnetics::IsolatedBuckBoost isolatedBuckBoostInputs(isolatedBuckBoostInputsJson);
        auto inputs = isolatedBuckBoostInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        
        // Add output voltages/currents to each operating point
        // These come from the original IsolatedBuckBoost operating points
        if (isolatedBuckBoostInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                // Extract the original operating point index from the result
                // The pattern is: for each input voltage, we iterate through all original operating points
                size_t originalOpIdx = opIdx % isolatedBuckBoostInputs.get_operating_points().size();
                auto origOp = isolatedBuckBoostInputs.get_operating_points()[originalOpIdx];
                
                // Create output waveforms similar to excitations
                // For DC outputs, we create constant waveforms at the output voltage/current levels
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    // Extract frequency and duty cycle from first excitation
                    double frequency = DEFAULT_FREQUENCY_HZ; // default
                    double dutyCycle = 0.5; // default
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        // Try to extract duty cycle from primary excitation
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;  // Switch on-time
                    double tOff = (1.0 - dutyCycle) * period;  // Switch off-time
                    int numPoints = 100;
                    
                    json voltageWaveform = {
                        {"time", json::array()},
                        {"data", json::array()}
                    };
                    
                    json currentWaveform = {
                        {"time", json::array()},
                        {"data", json::array()}
                    };
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    
                    // For primary output (index 0): create flyback-style pulsed current
                    // Current flows only during switch-off time (when diode conducts)
                    // Peak current is higher than DC to deliver same average power
                    if (i == 0) {
                        // Primary output current: zero during tOn, triangular ramp-down during tOff
                        double peakCurrent = outputCurrent / (1.0 - dutyCycle) * 2.0;  // Triangular average
                        
                        for (int j = 0; j < numPoints; j++) {
                            double t = (j / double(numPoints)) * period;
                            voltageWaveform["time"].push_back(t);
                            voltageWaveform["data"].push_back(outputVoltage);  // DC voltage
                            currentWaveform["time"].push_back(t);
                            
                            if (t < tOn) {
                                // Switch on: no output current (diode blocking)
                                currentWaveform["data"].push_back(0.0);
                            } else {
                                // Switch off: triangular ramp-down from peak to zero
                                double tInOffPeriod = t - tOn;
                                double current = peakCurrent * (1.0 - tInOffPeriod / tOff);
                                currentWaveform["data"].push_back(current);
                            }
                        }
                    } else {
                        // Secondary outputs (index 1+): also flyback-style pulsed current
                        double peakCurrent = outputCurrent / (1.0 - dutyCycle) * 2.0;
                        
                        for (int j = 0; j < numPoints; j++) {
                            double t = (j / double(numPoints)) * period;
                            voltageWaveform["time"].push_back(t);
                            voltageWaveform["data"].push_back(outputVoltage);  // DC voltage
                            currentWaveform["time"].push_back(t);
                            
                            if (t < tOn) {
                                currentWaveform["data"].push_back(0.0);
                            } else {
                                double tInOffPeriod = t - tOn;
                                double current = peakCurrent * (1.0 - tInOffPeriod / tOff);
                                currentWaveform["data"].push_back(current);
                            }
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }
        {
            json diag;
            const auto& names = isolatedBuckBoostInputs.get_per_op_name();
            const auto& v_duty_cycle = isolatedBuckBoostInputs.get_per_op_duty_cycle();
            const auto& v_magnetizing_current_ripple = isolatedBuckBoostInputs.get_per_op_magnetizing_current_ripple();
            const auto& v_primary_average_current = isolatedBuckBoostInputs.get_per_op_primary_average_current();
            const auto& v_primary_peak_current = isolatedBuckBoostInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = isolatedBuckBoostInputs.get_per_op_secondary_peak_current();
            const auto& v_is_ccm = isolatedBuckBoostInputs.get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? isolatedBuckBoostInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["magnetizingCurrentRipple"] = v_magnetizing_current_ripple.empty() ? isolatedBuckBoostInputs.get_last_magnetizing_current_ripple() : v_magnetizing_current_ripple.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? isolatedBuckBoostInputs.get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? isolatedBuckBoostInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? isolatedBuckBoostInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? isolatedBuckBoostInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["magnetizingCurrentRipple"] = v_magnetizing_current_ripple[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["isolatedBuckBoostDiagnostics"] = diag;
        }

        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_isolated_buck_boost_inputs(std::string isolatedBuckBoostInputsString){
    try {
        json isolatedBuckBoostInputsJson = json::parse(isolatedBuckBoostInputsString);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (isolatedBuckBoostInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = isolatedBuckBoostInputsJson["numberOfPeriods"].get<size_t>();
        }

        OpenMagnetics::AdvancedIsolatedBuckBoost isolatedBuckBoostInputs(isolatedBuckBoostInputsJson);
        auto inputs = isolatedBuckBoostInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        
        // Add output voltages/currents to each operating point
        if (isolatedBuckBoostInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % isolatedBuckBoostInputs.get_operating_points().size();
                auto origOp = isolatedBuckBoostInputs.get_operating_points()[originalOpIdx];
                
                // Create output waveforms similar to excitations
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    // Extract frequency and duty cycle from first excitation
                    double frequency = DEFAULT_FREQUENCY_HZ; // default
                    double dutyCycle = 0.5; // default
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    double tOff = (1.0 - dutyCycle) * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {
                        {"time", json::array()},
                        {"data", json::array()}
                    };
                    
                    json currentWaveform = {
                        {"time", json::array()},
                        {"data", json::array()}
                    };
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    
                    // Create flyback-style pulsed current for all outputs
                    double peakCurrent = outputCurrent / (1.0 - dutyCycle) * 2.0;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            currentWaveform["data"].push_back(0.0);
                        } else {
                            double tInOffPeriod = t - tOn;
                            double current = peakCurrent * (1.0 - tInOffPeriod / tOff);
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }
        {
            json diag;
            const auto& names = isolatedBuckBoostInputs.get_per_op_name();
            const auto& v_duty_cycle = isolatedBuckBoostInputs.get_per_op_duty_cycle();
            const auto& v_magnetizing_current_ripple = isolatedBuckBoostInputs.get_per_op_magnetizing_current_ripple();
            const auto& v_primary_average_current = isolatedBuckBoostInputs.get_per_op_primary_average_current();
            const auto& v_primary_peak_current = isolatedBuckBoostInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = isolatedBuckBoostInputs.get_per_op_secondary_peak_current();
            const auto& v_is_ccm = isolatedBuckBoostInputs.get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? isolatedBuckBoostInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["magnetizingCurrentRipple"] = v_magnetizing_current_ripple.empty() ? isolatedBuckBoostInputs.get_last_magnetizing_current_ripple() : v_magnetizing_current_ripple.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? isolatedBuckBoostInputs.get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? isolatedBuckBoostInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? isolatedBuckBoostInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? isolatedBuckBoostInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["magnetizingCurrentRipple"] = v_magnetizing_current_ripple[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["isolatedBuckBoostDiagnostics"] = diag;
        }

        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_buck_inputs(std::string buckInputsString){
    try {
        json buckInputsJson = json::parse(buckInputsString);

        OpenMagnetics::Buck buckInputs(buckInputsJson);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (buckInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = buckInputsJson["numberOfPeriods"].get<size_t>();
        }
        buckInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = buckInputs.process();

        json result;
        to_json(result, inputs);

        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = buckInputs.get_per_op_name();
            const auto& dC    = buckInputs.get_per_op_duty_cycle();
            const auto& iAvg  = buckInputs.get_per_op_inductor_average_current();
            const auto& iPP   = buckInputs.get_per_op_inductor_peak_to_peak();
            const auto& iPk   = buckInputs.get_per_op_peak_inductor_current();
            const auto& ccm   = buckInputs.get_per_op_is_ccm();
            const auto& cRat  = buckInputs.get_per_op_conduction_ratio();
            diag["dutyCycle"]              = dC.empty()   ? buckInputs.get_last_duty_cycle()              : dC.front();
            diag["inductorAverageCurrent"] = iAvg.empty() ? buckInputs.get_last_inductor_average_current() : iAvg.front();
            diag["inductorPeakToPeak"]     = iPP.empty()  ? buckInputs.get_last_inductor_peak_to_peak()    : iPP.front();
            diag["peakInductorCurrent"]    = iPk.empty()  ? buckInputs.get_last_peak_inductor_current()    : iPk.front();
            diag["conductionRatio"]        = cRat.empty() ? buckInputs.get_last_conduction_ratio()         : cRat.front();
            diag["isCcm"]                  = ccm.empty()  ? buckInputs.get_last_is_ccm()                   : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["inductorAverageCurrent"] = iAvg[i];
                row["inductorPeakToPeak"]    = iPP[i];
                row["peakInductorCurrent"]   = iPk[i];
                row["conductionRatio"]       = cRat[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["buckDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_buck_inputs(std::string buckInputsString){
    try {
        json buckInputsJson = json::parse(buckInputsString);

        OpenMagnetics::AdvancedBuck buckInputs(buckInputsJson);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (buckInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = buckInputsJson["numberOfPeriods"].get<size_t>();
        }
        buckInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = buckInputs.process();

        json result;
        to_json(result, inputs);

        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = buckInputs.get_per_op_name();
            const auto& dC    = buckInputs.get_per_op_duty_cycle();
            const auto& iAvg  = buckInputs.get_per_op_inductor_average_current();
            const auto& iPP   = buckInputs.get_per_op_inductor_peak_to_peak();
            const auto& iPk   = buckInputs.get_per_op_peak_inductor_current();
            const auto& ccm   = buckInputs.get_per_op_is_ccm();
            const auto& cRat  = buckInputs.get_per_op_conduction_ratio();
            diag["dutyCycle"]              = dC.empty()   ? buckInputs.get_last_duty_cycle()              : dC.front();
            diag["inductorAverageCurrent"] = iAvg.empty() ? buckInputs.get_last_inductor_average_current() : iAvg.front();
            diag["inductorPeakToPeak"]     = iPP.empty()  ? buckInputs.get_last_inductor_peak_to_peak()    : iPP.front();
            diag["peakInductorCurrent"]    = iPk.empty()  ? buckInputs.get_last_peak_inductor_current()    : iPk.front();
            diag["conductionRatio"]        = cRat.empty() ? buckInputs.get_last_conduction_ratio()         : cRat.front();
            diag["isCcm"]                  = ccm.empty()  ? buckInputs.get_last_is_ccm()                   : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["inductorAverageCurrent"] = iAvg[i];
                row["inductorPeakToPeak"]    = iPP[i];
                row["peakInductorCurrent"]   = iPk[i];
                row["conductionRatio"]       = cRat[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["buckDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_boost_inputs(std::string boostInputsString){
    try {
        json boostInputsJson = json::parse(boostInputsString);

        OpenMagnetics::Boost boostInputs(boostInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (boostInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = boostInputsJson["numberOfPeriods"].get<size_t>();
        }
        boostInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto inputs = boostInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = boostInputs.get_per_op_name();
            const auto& dC    = boostInputs.get_per_op_duty_cycle();
            const auto& iAvg  = boostInputs.get_per_op_inductor_average_current();
            const auto& iPP   = boostInputs.get_per_op_inductor_peak_to_peak();
            const auto& iPk   = boostInputs.get_per_op_peak_inductor_current();
            const auto& ccm   = boostInputs.get_per_op_is_ccm();
            const auto& cRat  = boostInputs.get_per_op_conduction_ratio();
            diag["dutyCycle"]              = dC.empty()   ? boostInputs.get_last_duty_cycle()              : dC.front();
            diag["inductorAverageCurrent"] = iAvg.empty() ? boostInputs.get_last_inductor_average_current() : iAvg.front();
            diag["inductorPeakToPeak"]     = iPP.empty()  ? boostInputs.get_last_inductor_peak_to_peak()    : iPP.front();
            diag["peakInductorCurrent"]    = iPk.empty()  ? boostInputs.get_last_peak_inductor_current()    : iPk.front();
            diag["conductionRatio"]        = cRat.empty() ? boostInputs.get_last_conduction_ratio()         : cRat.front();
            diag["isCcm"]                  = ccm.empty()  ? boostInputs.get_last_is_ccm()                   : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["inductorAverageCurrent"] = iAvg[i];
                row["inductorPeakToPeak"]    = iPP[i];
                row["peakInductorCurrent"]   = iPk[i];
                row["conductionRatio"]       = cRat[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["boostDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_boost_inputs(std::string boostInputsString){
    try {
        json boostInputsJson = json::parse(boostInputsString);

        OpenMagnetics::AdvancedBoost boostInputs(boostInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (boostInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = boostInputsJson["numberOfPeriods"].get<size_t>();
        }
        boostInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto inputs = boostInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = boostInputs.get_per_op_name();
            const auto& dC    = boostInputs.get_per_op_duty_cycle();
            const auto& iAvg  = boostInputs.get_per_op_inductor_average_current();
            const auto& iPP   = boostInputs.get_per_op_inductor_peak_to_peak();
            const auto& iPk   = boostInputs.get_per_op_peak_inductor_current();
            const auto& ccm   = boostInputs.get_per_op_is_ccm();
            const auto& cRat  = boostInputs.get_per_op_conduction_ratio();
            diag["dutyCycle"]              = dC.empty()   ? boostInputs.get_last_duty_cycle()              : dC.front();
            diag["inductorAverageCurrent"] = iAvg.empty() ? boostInputs.get_last_inductor_average_current() : iAvg.front();
            diag["inductorPeakToPeak"]     = iPP.empty()  ? boostInputs.get_last_inductor_peak_to_peak()    : iPP.front();
            diag["peakInductorCurrent"]    = iPk.empty()  ? boostInputs.get_last_peak_inductor_current()    : iPk.front();
            diag["conductionRatio"]        = cRat.empty() ? boostInputs.get_last_conduction_ratio()         : cRat.front();
            diag["isCcm"]                  = ccm.empty()  ? boostInputs.get_last_is_ccm()                   : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["inductorAverageCurrent"] = iAvg[i];
                row["inductorPeakToPeak"]    = iPP[i];
                row["peakInductorCurrent"]   = iPk[i];
                row["conductionRatio"]       = cRat[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["boostDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_sepic_inputs(std::string sepicInputsString){
    try {
        json sepicInputsJson = json::parse(sepicInputsString);

        OpenMagnetics::Sepic sepicInputs(sepicInputsJson);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (sepicInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = sepicInputsJson["numberOfPeriods"].get<size_t>();
        }
        sepicInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = sepicInputs.process();

        json result;
        to_json(result, inputs);

        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = sepicInputs.get_per_op_name();
            const auto& v_duty_cycle = sepicInputs.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = sepicInputs.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = sepicInputs.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = sepicInputs.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = sepicInputs.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = sepicInputs.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = sepicInputs.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = sepicInputs.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = sepicInputs.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = sepicInputs.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = sepicInputs.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = sepicInputs.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = sepicInputs.get_per_op_is_ccm();
            const auto& v_sized_cs = sepicInputs.get_per_op_sized_cs();
            const auto& v_sized_co = sepicInputs.get_per_op_sized_co();
            diag["dutyCycle"] = v_duty_cycle.empty() ? sepicInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? sepicInputs.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? sepicInputs.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? sepicInputs.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? sepicInputs.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? sepicInputs.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? sepicInputs.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? sepicInputs.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? sepicInputs.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? sepicInputs.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? sepicInputs.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? sepicInputs.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? sepicInputs.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCs"] = v_sized_cs.empty() ? sepicInputs.get_last_sized_cs() : v_sized_cs.front();
            diag["sizedCo"] = v_sized_co.empty() ? sepicInputs.get_last_sized_co() : v_sized_co.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCs"] = v_sized_cs[i];
                row["sizedCo"] = v_sized_co[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["sepicDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_sepic_inputs(std::string sepicInputsString){
    try {
        json sepicInputsJson = json::parse(sepicInputsString);

        OpenMagnetics::AdvancedSepic sepicInputs(sepicInputsJson);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (sepicInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = sepicInputsJson["numberOfPeriods"].get<size_t>();
        }
        sepicInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = sepicInputs.process();

        json result;
        to_json(result, inputs);

        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = sepicInputs.get_per_op_name();
            const auto& v_duty_cycle = sepicInputs.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = sepicInputs.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = sepicInputs.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = sepicInputs.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = sepicInputs.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = sepicInputs.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = sepicInputs.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = sepicInputs.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = sepicInputs.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = sepicInputs.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = sepicInputs.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = sepicInputs.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = sepicInputs.get_per_op_is_ccm();
            const auto& v_sized_cs = sepicInputs.get_per_op_sized_cs();
            const auto& v_sized_co = sepicInputs.get_per_op_sized_co();
            diag["dutyCycle"] = v_duty_cycle.empty() ? sepicInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? sepicInputs.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? sepicInputs.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? sepicInputs.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? sepicInputs.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? sepicInputs.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? sepicInputs.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? sepicInputs.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? sepicInputs.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? sepicInputs.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? sepicInputs.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? sepicInputs.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? sepicInputs.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCs"] = v_sized_cs.empty() ? sepicInputs.get_last_sized_cs() : v_sized_cs.front();
            diag["sizedCo"] = v_sized_co.empty() ? sepicInputs.get_last_sized_co() : v_sized_co.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCs"] = v_sized_cs[i];
                row["sizedCo"] = v_sized_co[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["sepicDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_buck_ideal_waveforms(std::string buckInputsString){
    try {
        json buckInputsJson = json::parse(buckInputsString);

        // Detect if this is an AdvancedBuck (user knows design) or regular Buck (help with design)
        bool isAdvancedBuck = buckInputsJson.contains("desiredInductance");
        
        DesignRequirements designRequirements;
        double inductance;
        
        std::unique_ptr<OpenMagnetics::Buck> buckPtr;
        
        if (isAdvancedBuck) {
            auto advancedBuckPtr = std::make_unique<OpenMagnetics::AdvancedBuck>(buckInputsJson);
            inductance = advancedBuckPtr->get_desired_inductance();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(inductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            std::vector<IsolationSide> isolationSides;
            isolationSides.push_back(OpenMagnetics::get_isolation_side_from_index(0));
            designRequirements.set_isolation_sides(isolationSides);
            designRequirements.set_topology(MAS::Topology::BUCK_CONVERTER);
            
            buckPtr = std::move(advancedBuckPtr);
        } else {
            buckPtr = std::make_unique<OpenMagnetics::Buck>(buckInputsJson);
            designRequirements = buckPtr->process_design_requirements();
            
            inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (buckInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = buckInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (buckInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = buckInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        // Set the number of periods to extract (not hardcoded to 1)
        buckPtr->set_num_periods_to_extract(numberOfPeriods);
        buckPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
        auto topologyWaveforms = buckPtr->simulate_and_extract_topology_waveforms(inductance);
        auto operatingPoints = buckPtr->simulate_and_extract_operating_points(inductance);

        // Build the result with just two fields: inputs and converterWaveforms
        json result;

        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;

        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: same schema as calculate_buck_inputs. The
        // simulate_and_extract_* path doesn't go through the analytical
        // code that sets last_* — populate them explicitly by running the
        // analytical pipeline once (Topology::process discards the result;
        // only the side-effect on last_* matters here).
        buckPtr->process();
        {
            json diag;
            const auto& names = buckPtr->get_per_op_name();
            const auto& dC    = buckPtr->get_per_op_duty_cycle();
            const auto& iAvg  = buckPtr->get_per_op_inductor_average_current();
            const auto& iPP   = buckPtr->get_per_op_inductor_peak_to_peak();
            const auto& iPk   = buckPtr->get_per_op_peak_inductor_current();
            const auto& ccm   = buckPtr->get_per_op_is_ccm();
            const auto& cRat  = buckPtr->get_per_op_conduction_ratio();
            diag["dutyCycle"]              = dC.empty()   ? buckPtr->get_last_duty_cycle()              : dC.front();
            diag["inductorAverageCurrent"] = iAvg.empty() ? buckPtr->get_last_inductor_average_current() : iAvg.front();
            diag["inductorPeakToPeak"]     = iPP.empty()  ? buckPtr->get_last_inductor_peak_to_peak()    : iPP.front();
            diag["peakInductorCurrent"]    = iPk.empty()  ? buckPtr->get_last_peak_inductor_current()    : iPk.front();
            diag["conductionRatio"]        = cRat.empty() ? buckPtr->get_last_conduction_ratio()         : cRat.front();
            diag["isCcm"]                  = ccm.empty()  ? buckPtr->get_last_is_ccm()                   : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["inductorAverageCurrent"] = iAvg[i];
                row["inductorPeakToPeak"]    = iPP[i];
                row["peakInductorCurrent"]   = iPk[i];
                row["conductionRatio"]       = cRat[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["buckDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_boost_ideal_waveforms(std::string boostInputsString){
    try {
        json boostInputsJson = json::parse(boostInputsString);

        // Detect if this is an AdvancedBoost (user knows design) or regular Boost (help with design)
        bool isAdvancedBoost = boostInputsJson.contains("desiredInductance");
        
        DesignRequirements designRequirements;
        double inductance;
        
        std::unique_ptr<OpenMagnetics::Boost> boostPtr;
        
        if (isAdvancedBoost) {
            auto advancedBoostPtr = std::make_unique<OpenMagnetics::AdvancedBoost>(boostInputsJson);
            inductance = advancedBoostPtr->get_desired_inductance();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(inductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            std::vector<IsolationSide> isolationSides;
            isolationSides.push_back(OpenMagnetics::get_isolation_side_from_index(0));
            designRequirements.set_isolation_sides(isolationSides);
            designRequirements.set_topology(MAS::Topology::BOOST_CONVERTER);
            
            boostPtr = std::move(advancedBoostPtr);
        } else {
            boostPtr = std::make_unique<OpenMagnetics::Boost>(boostInputsJson);
            designRequirements = boostPtr->process_design_requirements();
            
            inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (boostInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = boostInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (boostInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = boostInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        // Set the number of periods to extract (not hardcoded to 1)
        boostPtr->set_num_periods_to_extract(numberOfPeriods);
        boostPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
        auto topologyWaveforms = boostPtr->simulate_and_extract_topology_waveforms(inductance);
        auto operatingPoints = boostPtr->simulate_and_extract_operating_points(inductance);
        
        // DEBUG: Log operating point structure
        std::cerr << "DEBUG [simulate_boost]: Operating points count = " << operatingPoints.size() << std::endl;
        for (size_t opIdx = 0; opIdx < operatingPoints.size(); ++opIdx) {
            const auto& op = operatingPoints[opIdx];
            std::cerr << "  OP " << opIdx << ": excitations count = " << op.get_excitations_per_winding().size() << std::endl;
            for (size_t excIdx = 0; excIdx < op.get_excitations_per_winding().size(); ++excIdx) {
                const auto& exc = op.get_excitations_per_winding()[excIdx];
                std::cerr << "    Excitation " << excIdx << ":";
                if (exc.get_voltage() && exc.get_voltage()->get_waveform()) {
                    const auto vWf = exc.get_voltage()->get_waveform().value();  // Fix dangling reference
                    std::cerr << " voltage=" << vWf.get_data().size() << "pts";
                    if (!vWf.get_data().empty()) {
                        double minV = *std::min_element(vWf.get_data().begin(), vWf.get_data().end());
                        double maxV = *std::max_element(vWf.get_data().begin(), vWf.get_data().end());
                        std::cerr << " [" << minV << ".." << maxV << "V]";
                    }
                } else {
                    std::cerr << " voltage=none";
                }
                if (exc.get_current() && exc.get_current()->get_waveform()) {
                    const auto iWf = exc.get_current()->get_waveform().value();  // Fix dangling reference
                    std::cerr << " current=" << iWf.get_data().size() << "pts";
                    if (!iWf.get_data().empty()) {
                        double minI = *std::min_element(iWf.get_data().begin(), iWf.get_data().end());
                        double maxI = *std::max_element(iWf.get_data().begin(), iWf.get_data().end());
                        std::cerr << " [" << minI << ".." << maxI << "A]";
                    }
                } else {
                    std::cerr << " current=none";
                }
                std::cerr << std::endl;
            }
        }
         
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        boostPtr->process();
        {
            json diag;
            const auto& names = boostPtr->get_per_op_name();
            const auto& dC    = boostPtr->get_per_op_duty_cycle();
            const auto& iAvg  = boostPtr->get_per_op_inductor_average_current();
            const auto& iPP   = boostPtr->get_per_op_inductor_peak_to_peak();
            const auto& iPk   = boostPtr->get_per_op_peak_inductor_current();
            const auto& ccm   = boostPtr->get_per_op_is_ccm();
            const auto& cRat  = boostPtr->get_per_op_conduction_ratio();
            diag["dutyCycle"]              = dC.empty()   ? boostPtr->get_last_duty_cycle()              : dC.front();
            diag["inductorAverageCurrent"] = iAvg.empty() ? boostPtr->get_last_inductor_average_current() : iAvg.front();
            diag["inductorPeakToPeak"]     = iPP.empty()  ? boostPtr->get_last_inductor_peak_to_peak()    : iPP.front();
            diag["peakInductorCurrent"]    = iPk.empty()  ? boostPtr->get_last_peak_inductor_current()    : iPk.front();
            diag["conductionRatio"]        = cRat.empty() ? boostPtr->get_last_conduction_ratio()         : cRat.front();
            diag["isCcm"]                  = ccm.empty()  ? boostPtr->get_last_is_ccm()                   : (bool)ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < dC.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]             = dC[i];
                row["inductorAverageCurrent"] = iAvg[i];
                row["inductorPeakToPeak"]    = iPP[i];
                row["peakInductorCurrent"]   = iPk[i];
                row["conductionRatio"]       = cRat[i];
                row["isCcm"]                 = (bool)ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["boostDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_sepic_ideal_waveforms(std::string sepicInputsString){
    try {
        json sepicInputsJson = json::parse(sepicInputsString);

        // Detect if this is an AdvancedSepic (user knows design) or regular Sepic (help with design)
        bool isAdvancedSepic = sepicInputsJson.contains("desiredInductance");

        DesignRequirements designRequirements;
        double inductanceL1;

        std::unique_ptr<OpenMagnetics::Sepic> sepicPtr;

        if (isAdvancedSepic) {
            auto advancedSepicPtr = std::make_unique<OpenMagnetics::AdvancedSepic>(sepicInputsJson);
            inductanceL1 = advancedSepicPtr->get_desired_inductance();

            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(inductanceL1);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            std::vector<IsolationSide> isolationSides;
            isolationSides.push_back(OpenMagnetics::get_isolation_side_from_index(0));
            designRequirements.set_isolation_sides(isolationSides);
            designRequirements.set_topology(MAS::Topology::SEPIC_CONVERTER);

            sepicPtr = std::move(advancedSepicPtr);
        } else {
            sepicPtr = std::make_unique<OpenMagnetics::Sepic>(sepicInputsJson);
            designRequirements = sepicPtr->process_design_requirements();

            inductanceL1 = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductanceL1 > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif

        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (sepicInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = sepicInputsJson["numberOfPeriods"].get<size_t>();
        }

        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (sepicInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = sepicInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }

        // Set the number of periods to extract (not hardcoded to 1)
        sepicPtr->set_num_periods_to_extract(numberOfPeriods);
        sepicPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);

        auto topologyWaveforms = sepicPtr->simulate_and_extract_topology_waveforms(inductanceL1, numberOfPeriods);
        auto operatingPoints = sepicPtr->simulate_and_extract_operating_points(inductanceL1);

        // Build the result with just two fields: inputs and converterWaveforms
        json result;

        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;

        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        sepicPtr->process();
        {
            json diag;
            const auto& names = sepicPtr->get_per_op_name();
            const auto& v_duty_cycle = sepicPtr->get_per_op_duty_cycle();
            const auto& v_conversion_ratio = sepicPtr->get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = sepicPtr->get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = sepicPtr->get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = sepicPtr->get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = sepicPtr->get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = sepicPtr->get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = sepicPtr->get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = sepicPtr->get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = sepicPtr->get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = sepicPtr->get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = sepicPtr->get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = sepicPtr->get_per_op_is_ccm();
            const auto& v_sized_cs = sepicPtr->get_per_op_sized_cs();
            const auto& v_sized_co = sepicPtr->get_per_op_sized_co();
            diag["dutyCycle"] = v_duty_cycle.empty() ? sepicPtr->get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? sepicPtr->get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? sepicPtr->get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? sepicPtr->get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? sepicPtr->get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? sepicPtr->get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? sepicPtr->get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? sepicPtr->get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? sepicPtr->get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? sepicPtr->get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? sepicPtr->get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? sepicPtr->get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? sepicPtr->get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCs"] = v_sized_cs.empty() ? sepicPtr->get_last_sized_cs() : v_sized_cs.front();
            diag["sizedCo"] = v_sized_co.empty() ? sepicPtr->get_last_sized_co() : v_sized_co.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCs"] = v_sized_cs[i];
                row["sizedCo"] = v_sized_co[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["sepicDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_forward_ideal_waveforms(std::string forwardInputsString){
    try {
        json forwardInputsJson = json::parse(forwardInputsString);

        // Detect if this is an AdvancedSingleSwitchForward or regular
        bool isAdvanced = forwardInputsJson.contains("desiredInductance");
        
        // Read number of periods from input (default to 1)
        size_t numberOfPeriods = 1;
        if (forwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = forwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (forwardInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = forwardInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        DesignRequirements designRequirements;
        double magnetizingInductance;
        std::vector<double> turnsRatios;
        
        std::unique_ptr<OpenMagnetics::SingleSwitchForward> forwardPtr;
        
        if (isAdvanced) {
            auto advancedPtr = std::make_unique<OpenMagnetics::AdvancedSingleSwitchForward>(forwardInputsJson);
            magnetizingInductance = advancedPtr->get_desired_inductance();
            turnsRatios = advancedPtr->get_desired_turns_ratios();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto tr : turnsRatios) {
                DimensionWithTolerance trWithTolerance;
                trWithTolerance.set_nominal(tr);
                designRequirements.get_mutable_turns_ratios().push_back(trWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            designRequirements.set_topology(MAS::Topology::SINGLE_SWITCH_FORWARD_CONVERTER);
            
            forwardPtr = std::move(advancedPtr);
        } else {
            forwardPtr = std::make_unique<OpenMagnetics::SingleSwitchForward>(forwardInputsJson);
            designRequirements = forwardPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                if (tr.get_nominal()) {
                    turnsRatios.push_back(tr.get_nominal().value());
                }
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }
        
        forwardPtr->set_num_periods_to_extract(numberOfPeriods);
        forwardPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        auto topologyWaveforms = forwardPtr->simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance);
        auto operatingPoints = forwardPtr->simulate_and_extract_operating_points(turnsRatios, magnetizingInductance);
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        forwardPtr->process();
        {
            json diag;
            const auto& names = forwardPtr->get_per_op_name();
            const auto& v_maximum_duty_cycle = forwardPtr->get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = forwardPtr->get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = forwardPtr->get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = forwardPtr->get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = forwardPtr->get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = forwardPtr->get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = forwardPtr->get_per_op_is_ccm();
            const auto& v_computed_primary_turns_ratio = forwardPtr->get_per_op_computed_primary_turns_ratio();
            const auto& v_reset_voltage = forwardPtr->get_per_op_reset_voltage();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? forwardPtr->get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? forwardPtr->get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? forwardPtr->get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? forwardPtr->get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? forwardPtr->get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? forwardPtr->get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? forwardPtr->get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["primaryTurnsRatio"] = v_computed_primary_turns_ratio.empty() ? forwardPtr->get_last_computed_primary_turns_ratio() : v_computed_primary_turns_ratio.front();
            diag["resetVoltage"] = v_reset_voltage.empty() ? forwardPtr->get_last_reset_voltage() : v_reset_voltage.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["primaryTurnsRatio"] = v_computed_primary_turns_ratio[i];
                row["resetVoltage"] = v_reset_voltage[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["singleSwitchForwardDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_two_switch_forward_ideal_waveforms(std::string forwardInputsString){
    try {
        json forwardInputsJson = json::parse(forwardInputsString);

        // Detect if this is an AdvancedTwoSwitchForward or regular
        bool isAdvanced = forwardInputsJson.contains("desiredInductance");
        
        // Read number of periods from input (default to 1)
        size_t numberOfPeriods = 1;
        if (forwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = forwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (forwardInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = forwardInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        std::cerr << "DEBUG: TwoSwitchForward numberOfSteadyStatePeriods=" << numberOfSteadyStatePeriods << std::endl;
        
        DesignRequirements designRequirements;
        double magnetizingInductance;
        std::vector<double> turnsRatios;
        
        std::unique_ptr<OpenMagnetics::TwoSwitchForward> forwardPtr;
        
        if (isAdvanced) {
            auto advancedPtr = std::make_unique<OpenMagnetics::AdvancedTwoSwitchForward>(forwardInputsJson);
            magnetizingInductance = advancedPtr->get_desired_inductance();
            turnsRatios = advancedPtr->get_desired_turns_ratios();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto tr : turnsRatios) {
                DimensionWithTolerance trWithTolerance;
                trWithTolerance.set_nominal(tr);
                designRequirements.get_mutable_turns_ratios().push_back(trWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            designRequirements.set_topology(MAS::Topology::TWO_SWITCH_FORWARD_CONVERTER);
            
            forwardPtr = std::move(advancedPtr);
        } else {
            forwardPtr = std::make_unique<OpenMagnetics::TwoSwitchForward>(forwardInputsJson);
            designRequirements = forwardPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                if (tr.get_nominal()) {
                    turnsRatios.push_back(tr.get_nominal().value());
                }
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        }
        
        forwardPtr->set_num_periods_to_extract(numberOfPeriods);
        forwardPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        auto topologyWaveforms = forwardPtr->simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance);
        auto operatingPoints = forwardPtr->simulate_and_extract_operating_points(turnsRatios, magnetizingInductance);
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        forwardPtr->process();
        {
            json diag;
            const auto& names = forwardPtr->get_per_op_name();
            const auto& v_maximum_duty_cycle = forwardPtr->get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = forwardPtr->get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = forwardPtr->get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = forwardPtr->get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = forwardPtr->get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = forwardPtr->get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = forwardPtr->get_per_op_is_ccm();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? forwardPtr->get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? forwardPtr->get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? forwardPtr->get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? forwardPtr->get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? forwardPtr->get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? forwardPtr->get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? forwardPtr->get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["twoSwitchForwardDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_active_clamp_forward_ideal_waveforms(std::string forwardInputsString){
    try {
        json forwardInputsJson = json::parse(forwardInputsString);

        // Detect if this is an AdvancedActiveClampForward or regular
        bool isAdvanced = forwardInputsJson.contains("desiredInductance");
        
        // Read number of periods from input (default to 1)
        size_t numberOfPeriods = 1;
        if (forwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = forwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (forwardInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = forwardInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        DesignRequirements designRequirements;
        double magnetizingInductance;
        std::vector<double> turnsRatios;
        
        std::unique_ptr<OpenMagnetics::ActiveClampForward> forwardPtr;
        
        if (isAdvanced) {
            auto advancedPtr = std::make_unique<OpenMagnetics::AdvancedActiveClampForward>(forwardInputsJson);
            magnetizingInductance = advancedPtr->get_desired_inductance();
            turnsRatios = advancedPtr->get_desired_turns_ratios();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto tr : turnsRatios) {
                DimensionWithTolerance trWithTolerance;
                trWithTolerance.set_nominal(tr);
                designRequirements.get_mutable_turns_ratios().push_back(trWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            designRequirements.set_topology(MAS::Topology::ACTIVE_CLAMP_FORWARD_CONVERTER);
            
            forwardPtr = std::move(advancedPtr);
        } else {
            forwardPtr = std::make_unique<OpenMagnetics::ActiveClampForward>(forwardInputsJson);
            designRequirements = forwardPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                if (tr.get_nominal()) {
                    turnsRatios.push_back(tr.get_nominal().value());
                }
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        }
        
        forwardPtr->set_num_periods_to_extract(numberOfPeriods);
        forwardPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        auto topologyWaveforms = forwardPtr->simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance);
        auto operatingPoints = forwardPtr->simulate_and_extract_operating_points(turnsRatios, magnetizingInductance);
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        forwardPtr->process();
        {
            json diag;
            const auto& names = forwardPtr->get_per_op_name();
            const auto& v_maximum_duty_cycle = forwardPtr->get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = forwardPtr->get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = forwardPtr->get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = forwardPtr->get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = forwardPtr->get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = forwardPtr->get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = forwardPtr->get_per_op_is_ccm();
            const auto& v_clamp_cap_voltage = forwardPtr->get_per_op_clamp_cap_voltage();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? forwardPtr->get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? forwardPtr->get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? forwardPtr->get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? forwardPtr->get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? forwardPtr->get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? forwardPtr->get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? forwardPtr->get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["clampCapVoltage"] = v_clamp_cap_voltage.empty() ? forwardPtr->get_last_clamp_cap_voltage() : v_clamp_cap_voltage.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["clampCapVoltage"] = v_clamp_cap_voltage[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["activeClampForwardDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_push_pull_ideal_waveforms(std::string pushPullInputsString){
    try {
        json pushPullInputsJson = json::parse(pushPullInputsString);

        // Detect if this is an AdvancedPushPull or regular
        bool isAdvanced = pushPullInputsJson.contains("desiredInductance");
        
        // Debug: Log what was received for desiredTurnsRatios
        if (isAdvanced && pushPullInputsJson.contains("desiredTurnsRatios")) {
            std::cerr << "DEBUG [simulate_push_pull]: desiredTurnsRatios in JSON = " << pushPullInputsJson["desiredTurnsRatios"].dump() << std::endl;
        }
        
        // Read number of periods from input (default to 1)
        size_t numberOfPeriods = 1;
        if (pushPullInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = pushPullInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (pushPullInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = pushPullInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        DesignRequirements designRequirements;
        double magnetizingInductance;
        std::vector<double> turnsRatios;
        
        std::unique_ptr<OpenMagnetics::PushPull> pushPullPtr;
        
        if (isAdvanced) {
            auto advancedPtr = std::make_unique<OpenMagnetics::AdvancedPushPull>(pushPullInputsJson);
            magnetizingInductance = advancedPtr->get_desired_inductance();
            turnsRatios = advancedPtr->get_desired_turns_ratios();
            
            // Debug: Log extracted values
            std::cerr << "DEBUG [simulate_push_pull_ideal_waveforms]: Advanced mode" << std::endl;
            std::cerr << "  magnetizingInductance = " << magnetizingInductance << std::endl;
            std::cerr << "  turnsRatios.size() = " << turnsRatios.size() << std::endl;
            for (size_t i = 0; i < turnsRatios.size(); i++) {
                std::cerr << "  turnsRatios[" << i << "] = " << turnsRatios[i] << std::endl;
            }
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto tr : turnsRatios) {
                DimensionWithTolerance trWithTolerance;
                trWithTolerance.set_nominal(tr);
                designRequirements.get_mutable_turns_ratios().push_back(trWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            designRequirements.set_topology(MAS::Topology::PUSH_PULL_CONVERTER);
            
            pushPullPtr = std::move(advancedPtr);
        } else {
            pushPullPtr = std::make_unique<OpenMagnetics::PushPull>(pushPullInputsJson);
            designRequirements = pushPullPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                if (tr.get_nominal()) {
                    turnsRatios.push_back(tr.get_nominal().value());
                }
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }
        
        // Set steady state periods
        pushPullPtr->set_num_periods_to_extract(numberOfPeriods);
        pushPullPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        // CRITICAL FIX: Convert turns ratios from user format [N_sec/N_pri] to internal format [1, N_sec/N_pri, N_sec/N_pri, ...]
        // The ngspice netlist generation expects the internal format where:
        //   - index 0: second primary (always 1)
        //   - index 1,2: first and second secondary (same value for main output)
        //   - index 3+: auxiliary secondaries (if any)
        std::vector<double> convertedTurnsRatios;
        convertedTurnsRatios.push_back(1.0);  // Second primary
        convertedTurnsRatios.push_back(turnsRatios[0]);  // First secondary
        convertedTurnsRatios.push_back(turnsRatios[0]);  // Second secondary
        for (size_t i = 1; i < turnsRatios.size(); ++i) {
            convertedTurnsRatios.push_back(turnsRatios[i]);  // Auxiliary secondaries
        }
        
        std::cerr << "DEBUG: After convert_turns_ratios, size = " << convertedTurnsRatios.size() << std::endl;
        for (size_t i = 0; i < convertedTurnsRatios.size(); i++) {
            std::cerr << "  convertedTurnsRatios[" << i << "] = " << convertedTurnsRatios[i] << std::endl;
        }
        
        auto topologyWaveforms = pushPullPtr->simulate_and_extract_topology_waveforms(convertedTurnsRatios, magnetizingInductance);
        auto operatingPoints = pushPullPtr->simulate_and_extract_operating_points(convertedTurnsRatios, magnetizingInductance);
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        pushPullPtr->process();
        {
            json diag;
            const auto& names = pushPullPtr->get_per_op_name();
            const auto& v_duty_cycle = pushPullPtr->get_per_op_duty_cycle();
            const auto& v_switching_frequency = pushPullPtr->get_per_op_switching_frequency();
            const auto& v_primary_average_current = pushPullPtr->get_per_op_primary_average_current();
            const auto& v_primary_peak_current = pushPullPtr->get_per_op_primary_peak_current();
            const auto& v_magnetizing_peak_current = pushPullPtr->get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = pushPullPtr->get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? pushPullPtr->get_last_duty_cycle() : v_duty_cycle.front();
            diag["switchingFrequency"] = v_switching_frequency.empty() ? pushPullPtr->get_last_switching_frequency() : v_switching_frequency.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? pushPullPtr->get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? pushPullPtr->get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? pushPullPtr->get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? pushPullPtr->get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["switchingFrequency"] = v_switching_frequency[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pushPullDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_isolated_buck_boost_ideal_waveforms(std::string ibbInputsString){
    try {
        json ibbInputsJson = json::parse(ibbInputsString);

        // Detect if this is an AdvancedIsolatedBuckBoost or regular
        bool isAdvanced = ibbInputsJson.contains("desiredInductance");
        
        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (ibbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = ibbInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (ibbInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = ibbInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        DesignRequirements designRequirements;
        double magnetizingInductance;
        std::vector<double> turnsRatios;
        
        std::unique_ptr<OpenMagnetics::IsolatedBuckBoost> ibbPtr;
        
        if (isAdvanced) {
            auto advancedPtr = std::make_unique<OpenMagnetics::AdvancedIsolatedBuckBoost>(ibbInputsJson);
            magnetizingInductance = advancedPtr->get_desired_inductance();
            turnsRatios = advancedPtr->get_desired_turns_ratios();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto tr : turnsRatios) {
                DimensionWithTolerance trWithTolerance;
                trWithTolerance.set_nominal(tr);
                designRequirements.get_mutable_turns_ratios().push_back(trWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            designRequirements.set_topology(MAS::Topology::ISOLATED_BUCK_BOOST_CONVERTER);
            
            ibbPtr = std::move(advancedPtr);
        } else {
            ibbPtr = std::make_unique<OpenMagnetics::IsolatedBuckBoost>(ibbInputsJson);
            designRequirements = ibbPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                if (tr.get_nominal()) {
                    turnsRatios.push_back(tr.get_nominal().value());
                }
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }
        
        // Set steady state periods and extraction periods after pointer creation
        ibbPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        ibbPtr->set_num_periods_to_extract(numberOfPeriods);
        
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }
        
        auto topologyWaveforms = ibbPtr->simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance);
        auto operatingPoints = ibbPtr->simulate_and_extract_operating_points(turnsRatios, magnetizingInductance);
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        ibbPtr->process();
        {
            json diag;
            const auto& names = ibbPtr->get_per_op_name();
            const auto& v_duty_cycle = ibbPtr->get_per_op_duty_cycle();
            const auto& v_magnetizing_current_ripple = ibbPtr->get_per_op_magnetizing_current_ripple();
            const auto& v_primary_average_current = ibbPtr->get_per_op_primary_average_current();
            const auto& v_primary_peak_current = ibbPtr->get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = ibbPtr->get_per_op_secondary_peak_current();
            const auto& v_is_ccm = ibbPtr->get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? ibbPtr->get_last_duty_cycle() : v_duty_cycle.front();
            diag["magnetizingCurrentRipple"] = v_magnetizing_current_ripple.empty() ? ibbPtr->get_last_magnetizing_current_ripple() : v_magnetizing_current_ripple.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? ibbPtr->get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? ibbPtr->get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? ibbPtr->get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? ibbPtr->get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["magnetizingCurrentRipple"] = v_magnetizing_current_ripple[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["isolatedBuckBoostDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_isolated_buck_ideal_waveforms(std::string ibInputsString){
    try {
        json ibInputsJson = json::parse(ibInputsString);

        // Detect if this is an AdvancedIsolatedBuck or regular
        bool isAdvanced = ibInputsJson.contains("desiredInductance");
        
        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (ibInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = ibInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 5)
        size_t numberOfSteadyStatePeriods = 5;
        if (ibInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = ibInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        DesignRequirements designRequirements;
        double magnetizingInductance;
        std::vector<double> turnsRatios;
        
        std::unique_ptr<OpenMagnetics::IsolatedBuck> ibPtr;
        
        if (isAdvanced) {
            auto advancedPtr = std::make_unique<OpenMagnetics::AdvancedIsolatedBuck>(ibInputsJson);
            magnetizingInductance = advancedPtr->get_desired_inductance();
            turnsRatios = advancedPtr->get_desired_turns_ratios();
            
            // Build designRequirements
            designRequirements.get_mutable_turns_ratios().clear();
            for (auto tr : turnsRatios) {
                DimensionWithTolerance trWithTolerance;
                trWithTolerance.set_nominal(tr);
                designRequirements.get_mutable_turns_ratios().push_back(trWithTolerance);
            }
            DimensionWithTolerance inductanceWithTolerance;
            inductanceWithTolerance.set_nominal(magnetizingInductance);
            designRequirements.set_magnetizing_inductance(inductanceWithTolerance);
            designRequirements.set_topology(MAS::Topology::ISOLATED_BUCK_CONVERTER);
            
            ibPtr = std::move(advancedPtr);
        } else {
            ibPtr = std::make_unique<OpenMagnetics::IsolatedBuck>(ibInputsJson);
            designRequirements = ibPtr->process_design_requirements();
            
            // Extract turns ratios from design requirements
            for (const auto& tr : designRequirements.get_turns_ratios()) {
                if (tr.get_nominal()) {
                    turnsRatios.push_back(tr.get_nominal().value());
                }
            }
            
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("Unable to calculate inductance");
            }
        }
        
        // Set the number of periods for extraction
        ibPtr->set_num_periods_to_extract(numberOfPeriods);
        ibPtr->set_num_steady_state_periods(numberOfSteadyStatePeriods);
        
        // Get operating points from ngspice simulation (already has correct number of periods)
        auto operatingPoints = ibPtr->simulate_and_extract_operating_points(turnsRatios, magnetizingInductance);
        
        // DEBUG: Log operating point structure
        std::cerr << "DEBUG [simulate_isolated_buck]: Operating points count = " << operatingPoints.size() << std::endl;
        for (size_t opIdx = 0; opIdx < operatingPoints.size(); ++opIdx) {
            const auto& op = operatingPoints[opIdx];
            std::cerr << "  OP " << opIdx << ": excitations count = " << op.get_excitations_per_winding().size() << std::endl;
            for (size_t excIdx = 0; excIdx < op.get_excitations_per_winding().size(); ++excIdx) {
                const auto& exc = op.get_excitations_per_winding()[excIdx];
                std::cerr << "    Excitation " << excIdx << ":";
                if (exc.get_voltage() && exc.get_voltage()->get_waveform()) {
                    std::cerr << " voltage=" << exc.get_voltage()->get_waveform()->get_data().size() << "pts";
                } else {
                    std::cerr << " voltage=none";
                }
                if (exc.get_current() && exc.get_current()->get_waveform()) {
                    std::cerr << " current=" << exc.get_current()->get_waveform()->get_data().size() << "pts";
                } else {
                    std::cerr << " current=none";
                }
                std::cerr << std::endl;
            }
        }
        
        // Note: ngspice already extracted the correct number of periods, no need to repeat
        
        // Build the result with just two fields: inputs and converterWaveforms
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        auto topologyWaveforms = ibPtr->simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance);
        
        // DEBUG: Log converter waveforms structure
        std::cerr << "DEBUG [simulate_isolated_buck]: Topology waveforms count = " << topologyWaveforms.size() << std::endl;
        for (size_t twIdx = 0; twIdx < topologyWaveforms.size(); ++twIdx) {
            const auto& tw = topologyWaveforms[twIdx];
            std::cerr << "  TW " << twIdx << ":";
            if (tw.get_input_voltage().get_data().size() > 0) {
                std::cerr << " input_voltage=" << tw.get_input_voltage().get_data().size() << "pts";
            } else {
                std::cerr << " input_voltage=none";
            }
            std::cerr << " output_voltages=" << tw.get_output_voltages().size();
            std::cerr << " output_currents=" << tw.get_output_currents().size();
            std::cerr << std::endl;
            for (size_t outIdx = 0; outIdx < tw.get_output_voltages().size(); ++outIdx) {
                std::cerr << "    Output " << outIdx << ": voltage=" << tw.get_output_voltages()[outIdx].get_data().size() << "pts";
                if (outIdx < tw.get_output_currents().size()) {
                    std::cerr << " current=" << tw.get_output_currents()[outIdx].get_data().size() << "pts";
                }
                std::cerr << std::endl;
            }
        }
        
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Path B diagnostics: see Buck comment for rationale.
        ibPtr->process();
        {
            json diag;
            const auto& names = ibPtr->get_per_op_name();
            const auto& v_duty_cycle = ibPtr->get_per_op_duty_cycle();
            const auto& v_magnetizing_current_ripple = ibPtr->get_per_op_magnetizing_current_ripple();
            const auto& v_primary_average_current = ibPtr->get_per_op_primary_average_current();
            const auto& v_primary_peak_current = ibPtr->get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = ibPtr->get_per_op_secondary_peak_current();
            const auto& v_is_ccm = ibPtr->get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? ibPtr->get_last_duty_cycle() : v_duty_cycle.front();
            diag["magnetizingCurrentRipple"] = v_magnetizing_current_ripple.empty() ? ibPtr->get_last_magnetizing_current_ripple() : v_magnetizing_current_ripple.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? ibPtr->get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? ibPtr->get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? ibPtr->get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? ibPtr->get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["magnetizingCurrentRipple"] = v_magnetizing_current_ripple[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["isolatedBuckDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_push_pull_inputs(std::string pushPullInputsString){
    try {
        json pushPullInputsJson = json::parse(pushPullInputsString);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (pushPullInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = pushPullInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Debug: Log current ripple ratio
        std::cerr << "DEBUG [calculate_push_pull_inputs]: currentRippleRatio = ";
        if (pushPullInputsJson.contains("currentRippleRatio")) {
            std::cerr << pushPullInputsJson["currentRippleRatio"].get<double>();
        } else {
            std::cerr << "not set";
        }
        std::cerr << std::endl;

        OpenMagnetics::PushPull pushPullInputs(pushPullInputsJson);
        auto inputs = pushPullInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = pushPullInputs.get_per_op_name();
            const auto& v_duty_cycle = pushPullInputs.get_per_op_duty_cycle();
            const auto& v_switching_frequency = pushPullInputs.get_per_op_switching_frequency();
            const auto& v_primary_average_current = pushPullInputs.get_per_op_primary_average_current();
            const auto& v_primary_peak_current = pushPullInputs.get_per_op_primary_peak_current();
            const auto& v_magnetizing_peak_current = pushPullInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = pushPullInputs.get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? pushPullInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["switchingFrequency"] = v_switching_frequency.empty() ? pushPullInputs.get_last_switching_frequency() : v_switching_frequency.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? pushPullInputs.get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? pushPullInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? pushPullInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? pushPullInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["switchingFrequency"] = v_switching_frequency[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pushPullDiagnostics"] = diag;
        }

        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << "ERROR [calculate_push_pull_inputs]: " << exc.what() << std::endl;
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_advanced_push_pull_inputs(std::string pushPullInputsString){
    try {
        json pushPullInputsJson = json::parse(pushPullInputsString);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (pushPullInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = pushPullInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Debug: Log received parameters
        std::cerr << "DEBUG [calculate_advanced_push_pull_inputs]: Received JSON with keys: ";
        for (auto& el : pushPullInputsJson.items()) {
            std::cerr << el.key() << ", ";
        }
        std::cerr << std::endl;
        
        // Debug: Log the critical values
        if (pushPullInputsJson.contains("desiredTurnsRatios")) {
            std::cerr << "  desiredTurnsRatios = " << pushPullInputsJson["desiredTurnsRatios"].dump() << std::endl;
        }
        if (pushPullInputsJson.contains("desiredDutyCycle")) {
            std::cerr << "  desiredDutyCycle = " << pushPullInputsJson["desiredDutyCycle"].dump() << std::endl;
        }
        if (pushPullInputsJson.contains("desiredInductance")) {
            std::cerr << "  desiredInductance = " << pushPullInputsJson["desiredInductance"] << std::endl;
        }
        if (pushPullInputsJson.contains("inputVoltage")) {
            std::cerr << "  inputVoltage = " << pushPullInputsJson["inputVoltage"].dump() << std::endl;
        }
        
        // IMPORTANT: The analytical process() method expects user-format turns ratios [N]
        // and internally converts them to [1, N, N, ...] for calculations.
        // So we do NOT convert here - just pass through as-is.
        // The JSON already has the correct format from the frontend.

        OpenMagnetics::AdvancedPushPull pushPullInputs(pushPullInputsJson);
        auto inputs = pushPullInputs.process();

        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = pushPullInputs.get_per_op_name();
            const auto& v_duty_cycle = pushPullInputs.get_per_op_duty_cycle();
            const auto& v_switching_frequency = pushPullInputs.get_per_op_switching_frequency();
            const auto& v_primary_average_current = pushPullInputs.get_per_op_primary_average_current();
            const auto& v_primary_peak_current = pushPullInputs.get_per_op_primary_peak_current();
            const auto& v_magnetizing_peak_current = pushPullInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = pushPullInputs.get_per_op_is_ccm();
            diag["dutyCycle"] = v_duty_cycle.empty() ? pushPullInputs.get_last_duty_cycle() : v_duty_cycle.front();
            diag["switchingFrequency"] = v_switching_frequency.empty() ? pushPullInputs.get_last_switching_frequency() : v_switching_frequency.front();
            diag["primaryAverageCurrent"] = v_primary_average_current.empty() ? pushPullInputs.get_last_primary_average_current() : v_primary_average_current.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? pushPullInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? pushPullInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? pushPullInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["switchingFrequency"] = v_switching_frequency[i];
                row["primaryAverageCurrent"] = v_primary_average_current[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pushPullDiagnostics"] = diag;
        }

        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        std::cerr << "ERROR [calculate_advanced_push_pull_inputs]: " << exc.what() << std::endl;
        json error;
        error["error"] = true;
        error["message"] = exc.what();
        return error.dump(4);
    }
}

std::string calculate_single_switch_forward_inputs(std::string singleSwitchForwardInputsString){
    try {
        json singleSwitchForwardInputsJson = json::parse(singleSwitchForwardInputsString);

        OpenMagnetics::SingleSwitchForward singleSwitchForwardInputs(singleSwitchForwardInputsJson);

        // Read number of periods from input
        size_t numberOfPeriods = 2;
        if (singleSwitchForwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = singleSwitchForwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        singleSwitchForwardInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = singleSwitchForwardInputs.process();

        json result;
        to_json(result, inputs);
        
        // Add output voltages/currents to each operating point
        if (singleSwitchForwardInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % singleSwitchForwardInputs.get_operating_points().size();
                auto origOp = singleSwitchForwardInputs.get_operating_points()[originalOpIdx];
                
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    double frequency = DEFAULT_FREQUENCY_HZ;
                    double dutyCycle = 0.5;
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {{"time", json::array()}, {"data", json::array()}};
                    json currentWaveform = {{"time", json::array()}, {"data", json::array()}};
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    double currentRipple = outputCurrent * 0.3;
                    double minCurrent = outputCurrent - currentRipple / 2;
                    double maxCurrent = outputCurrent + currentRipple / 2;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            double progress = t / tOn;
                            double current = minCurrent + (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        } else {
                            double progress = (t - tOn) / (period - tOn);
                            double current = maxCurrent - (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }

        // Repeat waveforms for the requested number of periods
        if (result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = singleSwitchForwardInputs.get_per_op_name();
            const auto& v_maximum_duty_cycle = singleSwitchForwardInputs.get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = singleSwitchForwardInputs.get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = singleSwitchForwardInputs.get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = singleSwitchForwardInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = singleSwitchForwardInputs.get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = singleSwitchForwardInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = singleSwitchForwardInputs.get_per_op_is_ccm();
            const auto& v_computed_primary_turns_ratio = singleSwitchForwardInputs.get_per_op_computed_primary_turns_ratio();
            const auto& v_reset_voltage = singleSwitchForwardInputs.get_per_op_reset_voltage();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? singleSwitchForwardInputs.get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? singleSwitchForwardInputs.get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? singleSwitchForwardInputs.get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? singleSwitchForwardInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? singleSwitchForwardInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? singleSwitchForwardInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? singleSwitchForwardInputs.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["primaryTurnsRatio"] = v_computed_primary_turns_ratio.empty() ? singleSwitchForwardInputs.get_last_computed_primary_turns_ratio() : v_computed_primary_turns_ratio.front();
            diag["resetVoltage"] = v_reset_voltage.empty() ? singleSwitchForwardInputs.get_last_reset_voltage() : v_reset_voltage.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["primaryTurnsRatio"] = v_computed_primary_turns_ratio[i];
                row["resetVoltage"] = v_reset_voltage[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["singleSwitchForwardDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_single_switch_forward_inputs(std::string singleSwitchForwardInputsString){
    try {
        json singleSwitchForwardInputsJson = json::parse(singleSwitchForwardInputsString);

        OpenMagnetics::AdvancedSingleSwitchForward singleSwitchForwardInputs(singleSwitchForwardInputsJson);

        // Read number of periods from input
        size_t numberOfPeriods = 2;
        if (singleSwitchForwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = singleSwitchForwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        singleSwitchForwardInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = singleSwitchForwardInputs.process();

        json result;
        to_json(result, inputs);
        
        // Add output voltages/currents to each operating point
        if (singleSwitchForwardInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % singleSwitchForwardInputs.get_operating_points().size();
                auto origOp = singleSwitchForwardInputs.get_operating_points()[originalOpIdx];
                
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    double frequency = DEFAULT_FREQUENCY_HZ;
                    double dutyCycle = 0.5;
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {{"time", json::array()}, {"data", json::array()}};
                    json currentWaveform = {{"time", json::array()}, {"data", json::array()}};
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    double currentRipple = outputCurrent * 0.3;
                    double minCurrent = outputCurrent - currentRipple / 2;
                    double maxCurrent = outputCurrent + currentRipple / 2;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            double progress = t / tOn;
                            double current = minCurrent + (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        } else {
                            double progress = (t - tOn) / (period - tOn);
                            double current = maxCurrent - (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }

        // Repeat waveforms for the requested number of periods
        if (result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = singleSwitchForwardInputs.get_per_op_name();
            const auto& v_maximum_duty_cycle = singleSwitchForwardInputs.get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = singleSwitchForwardInputs.get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = singleSwitchForwardInputs.get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = singleSwitchForwardInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = singleSwitchForwardInputs.get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = singleSwitchForwardInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = singleSwitchForwardInputs.get_per_op_is_ccm();
            const auto& v_computed_primary_turns_ratio = singleSwitchForwardInputs.get_per_op_computed_primary_turns_ratio();
            const auto& v_reset_voltage = singleSwitchForwardInputs.get_per_op_reset_voltage();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? singleSwitchForwardInputs.get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? singleSwitchForwardInputs.get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? singleSwitchForwardInputs.get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? singleSwitchForwardInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? singleSwitchForwardInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? singleSwitchForwardInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? singleSwitchForwardInputs.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["primaryTurnsRatio"] = v_computed_primary_turns_ratio.empty() ? singleSwitchForwardInputs.get_last_computed_primary_turns_ratio() : v_computed_primary_turns_ratio.front();
            diag["resetVoltage"] = v_reset_voltage.empty() ? singleSwitchForwardInputs.get_last_reset_voltage() : v_reset_voltage.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["primaryTurnsRatio"] = v_computed_primary_turns_ratio[i];
                row["resetVoltage"] = v_reset_voltage[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["singleSwitchForwardDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error = {
            {"error", true},
            {"message", std::string{exc.what()}}
        };
        return error.dump(4);
    }
}

std::string calculate_active_clamp_forward_inputs(std::string activeClampForwardInputsString){
    try {
        json activeClampForwardInputsJson = json::parse(activeClampForwardInputsString);

        OpenMagnetics::ActiveClampForward activeClampForwardInputs(activeClampForwardInputsJson);

        // Read number of periods from input
        size_t numberOfPeriods = 2;
        if (activeClampForwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = activeClampForwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        activeClampForwardInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = activeClampForwardInputs.process();

        json result;
        to_json(result, inputs);
        
        // Add output voltages/currents to each operating point
        if (activeClampForwardInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % activeClampForwardInputs.get_operating_points().size();
                auto origOp = activeClampForwardInputs.get_operating_points()[originalOpIdx];
                
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    double frequency = DEFAULT_FREQUENCY_HZ;
                    double dutyCycle = 0.5;
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {{"time", json::array()}, {"data", json::array()}};
                    json currentWaveform = {{"time", json::array()}, {"data", json::array()}};
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    double currentRipple = outputCurrent * 0.3;
                    double minCurrent = outputCurrent - currentRipple / 2;
                    double maxCurrent = outputCurrent + currentRipple / 2;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            double progress = t / tOn;
                            double current = minCurrent + (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        } else {
                            double progress = (t - tOn) / (period - tOn);
                            double current = maxCurrent - (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }

        // Repeat waveforms for the requested number of periods
        if (result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = activeClampForwardInputs.get_per_op_name();
            const auto& v_maximum_duty_cycle = activeClampForwardInputs.get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = activeClampForwardInputs.get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = activeClampForwardInputs.get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = activeClampForwardInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = activeClampForwardInputs.get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = activeClampForwardInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = activeClampForwardInputs.get_per_op_is_ccm();
            const auto& v_clamp_cap_voltage = activeClampForwardInputs.get_per_op_clamp_cap_voltage();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? activeClampForwardInputs.get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? activeClampForwardInputs.get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? activeClampForwardInputs.get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? activeClampForwardInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? activeClampForwardInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? activeClampForwardInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? activeClampForwardInputs.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["clampCapVoltage"] = v_clamp_cap_voltage.empty() ? activeClampForwardInputs.get_last_clamp_cap_voltage() : v_clamp_cap_voltage.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["clampCapVoltage"] = v_clamp_cap_voltage[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["activeClampForwardDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_active_clamp_forward_inputs(std::string activeClampForwardInputsString){
    try {
        json activeClampForwardInputsJson = json::parse(activeClampForwardInputsString);

        OpenMagnetics::AdvancedActiveClampForward activeClampForwardInputs(activeClampForwardInputsJson);

        // Read number of periods from input
        size_t numberOfPeriods = 2;
        if (activeClampForwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = activeClampForwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        activeClampForwardInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = activeClampForwardInputs.process();

        json result;
        to_json(result, inputs);
        
        // Add output voltages/currents to each operating point
        if (activeClampForwardInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % activeClampForwardInputs.get_operating_points().size();
                auto origOp = activeClampForwardInputs.get_operating_points()[originalOpIdx];
                
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    double frequency = DEFAULT_FREQUENCY_HZ;
                    double dutyCycle = 0.5;
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {{"time", json::array()}, {"data", json::array()}};
                    json currentWaveform = {{"time", json::array()}, {"data", json::array()}};
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    double currentRipple = outputCurrent * 0.3;
                    double minCurrent = outputCurrent - currentRipple / 2;
                    double maxCurrent = outputCurrent + currentRipple / 2;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            double progress = t / tOn;
                            double current = minCurrent + (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        } else {
                            double progress = (t - tOn) / (period - tOn);
                            double current = maxCurrent - (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }

        // Repeat waveforms for the requested number of periods
        if (result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = activeClampForwardInputs.get_per_op_name();
            const auto& v_maximum_duty_cycle = activeClampForwardInputs.get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = activeClampForwardInputs.get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = activeClampForwardInputs.get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = activeClampForwardInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = activeClampForwardInputs.get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = activeClampForwardInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = activeClampForwardInputs.get_per_op_is_ccm();
            const auto& v_clamp_cap_voltage = activeClampForwardInputs.get_per_op_clamp_cap_voltage();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? activeClampForwardInputs.get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? activeClampForwardInputs.get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? activeClampForwardInputs.get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? activeClampForwardInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? activeClampForwardInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? activeClampForwardInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? activeClampForwardInputs.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["clampCapVoltage"] = v_clamp_cap_voltage.empty() ? activeClampForwardInputs.get_last_clamp_cap_voltage() : v_clamp_cap_voltage.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["clampCapVoltage"] = v_clamp_cap_voltage[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["activeClampForwardDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error = {
            {"error", true},
            {"message", std::string{exc.what()}}
        };
        return error.dump(4);
    }
}

std::string calculate_two_switch_forward_inputs(std::string twoSwitchForwardInputsString){
    try {
        json twoSwitchForwardInputsJson = json::parse(twoSwitchForwardInputsString);

        OpenMagnetics::TwoSwitchForward twoSwitchForwardInputs(twoSwitchForwardInputsJson);

        // Read number of periods from input
        size_t numberOfPeriods = 2;
        if (twoSwitchForwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = twoSwitchForwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        twoSwitchForwardInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = twoSwitchForwardInputs.process();

        json result;
        to_json(result, inputs);
        
        // Add output voltages/currents to each operating point
        if (twoSwitchForwardInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % twoSwitchForwardInputs.get_operating_points().size();
                auto origOp = twoSwitchForwardInputs.get_operating_points()[originalOpIdx];
                
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    double frequency = DEFAULT_FREQUENCY_HZ;
                    double dutyCycle = 0.5;
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {{"time", json::array()}, {"data", json::array()}};
                    json currentWaveform = {{"time", json::array()}, {"data", json::array()}};
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    double currentRipple = outputCurrent * 0.3;
                    double minCurrent = outputCurrent - currentRipple / 2;
                    double maxCurrent = outputCurrent + currentRipple / 2;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            double progress = t / tOn;
                            double current = minCurrent + (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        } else {
                            double progress = (t - tOn) / (period - tOn);
                            double current = maxCurrent - (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }

        // Repeat waveforms for the requested number of periods
        if (result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = twoSwitchForwardInputs.get_per_op_name();
            const auto& v_maximum_duty_cycle = twoSwitchForwardInputs.get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = twoSwitchForwardInputs.get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = twoSwitchForwardInputs.get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = twoSwitchForwardInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = twoSwitchForwardInputs.get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = twoSwitchForwardInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = twoSwitchForwardInputs.get_per_op_is_ccm();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? twoSwitchForwardInputs.get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? twoSwitchForwardInputs.get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? twoSwitchForwardInputs.get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? twoSwitchForwardInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? twoSwitchForwardInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? twoSwitchForwardInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? twoSwitchForwardInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["twoSwitchForwardDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_two_switch_forward_inputs(std::string twoSwitchForwardInputsString){
    try {
        json twoSwitchForwardInputsJson = json::parse(twoSwitchForwardInputsString);

        OpenMagnetics::AdvancedTwoSwitchForward twoSwitchForwardInputs(twoSwitchForwardInputsJson);

        // Read number of periods from input
        size_t numberOfPeriods = 2;
        if (twoSwitchForwardInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = twoSwitchForwardInputsJson["numberOfPeriods"].get<size_t>();
        }
        twoSwitchForwardInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = twoSwitchForwardInputs.process();

        json result;
        to_json(result, inputs);
        
        // Add output voltages/currents to each operating point
        if (twoSwitchForwardInputs.get_operating_points().size() > 0 && result.contains("operatingPoints")) {
            size_t opIdx = 0;
            for (auto& op : result["operatingPoints"]) {
                size_t originalOpIdx = opIdx % twoSwitchForwardInputs.get_operating_points().size();
                auto origOp = twoSwitchForwardInputs.get_operating_points()[originalOpIdx];
                
                json outputVoltagesArray = json::array();
                json outputCurrentsArray = json::array();
                
                for (size_t i = 0; i < origOp.get_output_voltages().size(); i++) {
                    double frequency = DEFAULT_FREQUENCY_HZ;
                    double dutyCycle = 0.5;
                    if (op.contains("excitationsPerWinding") && op["excitationsPerWinding"].size() > 0) {
                        if (op["excitationsPerWinding"][0].contains("frequency")) {
                            frequency = op["excitationsPerWinding"][0]["frequency"];
                        }
                        if (op["excitationsPerWinding"][0].contains("current") && 
                            op["excitationsPerWinding"][0]["current"].contains("processed") &&
                            op["excitationsPerWinding"][0]["current"]["processed"].contains("dutyCycle")) {
                            dutyCycle = op["excitationsPerWinding"][0]["current"]["processed"]["dutyCycle"];
                        }
                    }
                    
                    double period = 1.0 / frequency;
                    double tOn = dutyCycle * period;
                    int numPoints = 100;
                    
                    json voltageWaveform = {{"time", json::array()}, {"data", json::array()}};
                    json currentWaveform = {{"time", json::array()}, {"data", json::array()}};
                    
                    double outputVoltage = origOp.get_output_voltages()[i];
                    double outputCurrent = origOp.get_output_currents()[i];
                    double currentRipple = outputCurrent * 0.3;
                    double minCurrent = outputCurrent - currentRipple / 2;
                    double maxCurrent = outputCurrent + currentRipple / 2;
                    
                    for (int j = 0; j < numPoints; j++) {
                        double t = (j / double(numPoints)) * period;
                        voltageWaveform["time"].push_back(t);
                        voltageWaveform["data"].push_back(outputVoltage);
                        currentWaveform["time"].push_back(t);
                        
                        if (t < tOn) {
                            double progress = t / tOn;
                            double current = minCurrent + (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        } else {
                            double progress = (t - tOn) / (period - tOn);
                            double current = maxCurrent - (maxCurrent - minCurrent) * progress;
                            currentWaveform["data"].push_back(current);
                        }
                    }
                    
                    outputVoltagesArray.push_back({{"waveform", voltageWaveform}});
                    outputCurrentsArray.push_back({{"waveform", currentWaveform}});
                }
                
                op["outputVoltages"] = outputVoltagesArray;
                op["outputCurrents"] = outputCurrentsArray;
                opIdx++;
            }
        }

        // Repeat waveforms for the requested number of periods
        if (result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = twoSwitchForwardInputs.get_per_op_name();
            const auto& v_maximum_duty_cycle = twoSwitchForwardInputs.get_per_op_maximum_duty_cycle();
            const auto& v_computed_magnetizing_inductance = twoSwitchForwardInputs.get_per_op_computed_magnetizing_inductance();
            const auto& v_computed_secondary_turns_ratio = twoSwitchForwardInputs.get_per_op_computed_secondary_turns_ratio();
            const auto& v_primary_peak_current = twoSwitchForwardInputs.get_per_op_primary_peak_current();
            const auto& v_secondary_peak_current = twoSwitchForwardInputs.get_per_op_secondary_peak_current();
            const auto& v_magnetizing_peak_current = twoSwitchForwardInputs.get_per_op_magnetizing_peak_current();
            const auto& v_is_ccm = twoSwitchForwardInputs.get_per_op_is_ccm();
            diag["maximumDutyCycle"] = v_maximum_duty_cycle.empty() ? twoSwitchForwardInputs.get_last_maximum_duty_cycle() : v_maximum_duty_cycle.front();
            diag["magnetizingInductance"] = v_computed_magnetizing_inductance.empty() ? twoSwitchForwardInputs.get_last_computed_magnetizing_inductance() : v_computed_magnetizing_inductance.front();
            diag["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio.empty() ? twoSwitchForwardInputs.get_last_computed_secondary_turns_ratio() : v_computed_secondary_turns_ratio.front();
            diag["primaryPeakCurrent"] = v_primary_peak_current.empty() ? twoSwitchForwardInputs.get_last_primary_peak_current() : v_primary_peak_current.front();
            diag["secondaryPeakCurrent"] = v_secondary_peak_current.empty() ? twoSwitchForwardInputs.get_last_secondary_peak_current() : v_secondary_peak_current.front();
            diag["magnetizingPeakCurrent"] = v_magnetizing_peak_current.empty() ? twoSwitchForwardInputs.get_last_magnetizing_peak_current() : v_magnetizing_peak_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? twoSwitchForwardInputs.get_last_is_ccm() : (bool)v_is_ccm.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_maximum_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["maximumDutyCycle"] = v_maximum_duty_cycle[i];
                row["magnetizingInductance"] = v_computed_magnetizing_inductance[i];
                row["secondaryTurnsRatio"] = v_computed_secondary_turns_ratio[i];
                row["primaryPeakCurrent"] = v_primary_peak_current[i];
                row["secondaryPeakCurrent"] = v_secondary_peak_current[i];
                row["magnetizingPeakCurrent"] = v_magnetizing_peak_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["twoSwitchForwardDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error = {
            {"error", true},
            {"message", std::string{exc.what()}}
        };
        return error.dump(4);
    }
}

// ==========================================
// Power Factor Correction (PFC) Functions
// ==========================================

EMSCRIPTEN_KEEPALIVE std::string calculate_pfc_inputs(std::string pfcInputsString){
    try {
        json pfcInputsJson = json::parse(pfcInputsString);

        OpenMagnetics::PowerFactorCorrection pfcInputs(pfcInputsJson);
        
        // Set number of periods for waveform generation
        int numberOfPeriods = pfcInputsJson.value("numberOfPeriods", 2);
        // pfcInputs.set_number_of_periods(numberOfPeriods);
        
        auto designRequirements = pfcInputs.process_design_requirements();
        
        // Calculate inductance based on mode
        double inductance;
        if (pfcInputsJson.contains("inductance")) {
            inductance = pfcInputsJson["inductance"];
        } else {
            auto mode = pfcInputs.get_mode().value_or(PfcModes::CONTINUOUS_CONDUCTION_MODE);
            switch (mode) {
                case PfcModes::CONTINUOUS_CONDUCTION_MODE:
                    inductance = pfcInputs.calculate_inductance_ccm(); break;
                case PfcModes::CRITICAL_CONDUCTION_MODE:
                    inductance = pfcInputs.calculate_inductance_crcm(); break;
                case PfcModes::DISCONTINUOUS_CONDUCTION_MODE:
                    inductance = pfcInputs.calculate_inductance_dcm(); break;
                default:
                    throw std::runtime_error("Unsupported PFC mode for calculate_pfc_inputs");
            }
        }
        
        // Get operating points at different AC line phases (single period for MAS)
        auto operatingPoints = pfcInputs.process_operating_points({}, inductance);
        
        // Get full multi-period waveforms for display (respects numberOfPeriods setting)
        // Note: simulate_and_extract_waveforms already generates the requested number of periods
        auto displayWaveforms = pfcInputs.simulate_and_extract_waveforms(inductance, 0.1, numberOfPeriods);
        
        // Build result with inputs and waveforms.
        // Emit designRequirements as a NESTED object (matching calculate_sepic_inputs
        // and every other converter entry) so ConverterWizardBase captures the full
        // MAS DesignRequirements. Spreading the fields at top level left
        // result.designRequirements undefined, forcing the wizard onto a skeleton DR.
        json result;
        to_json(result["designRequirements"], designRequirements);
        result["inductance"] = inductance;
        result["operatingPoints"] = json::array();
        
        for (auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            result["operatingPoints"].push_back(opJson);
        }
        
        // Note: No need to call repeat_operating_points_waveforms here because
        // simulate_and_extract_waveforms already generated the correct number of periods
        
        // Build MAS inputs - only include first operating point for magnetic design.
        // designRequirements is nested under its own key so masInputs is a valid
        // MAS Inputs object (designRequirements + operatingPoints).
        result["masInputs"] = json::object();
        to_json(result["masInputs"]["designRequirements"], designRequirements);
        // Only export first operating point to magnetic tool to avoid redundant calculations
        if (!result["operatingPoints"].empty()) {
            result["masInputs"]["operatingPoints"] = json::array({result["operatingPoints"][0]});
        } else {
            result["masInputs"]["operatingPoints"] = json::array();
        }

        {
            json diag;
            diag["computedInductance"]      = pfcInputs.get_computed_inductance();
            diag["actualMode"]              = pfcInputs.get_computed_actual_mode();
            const auto& names = pfcInputs.get_per_op_name();
            const auto& dc    = pfcInputs.get_per_op_duty_cycle_peak();
            const auto& ipk   = pfcInputs.get_per_op_peak_inductor_current();
            const auto& ir    = pfcInputs.get_per_op_inductor_ripple();
            const auto& irms  = pfcInputs.get_per_op_line_rms_current();
            const auto& pin   = pfcInputs.get_per_op_input_power();
            diag["dutyCyclePeak"]           = dc.empty()   ? pfcInputs.get_last_duty_cycle_peak()         : dc.front();
            diag["peakInductorCurrent"]     = ipk.empty()  ? pfcInputs.get_last_peak_inductor_current()   : ipk.front();
            diag["inductorRipple"]          = ir.empty()   ? pfcInputs.get_last_inductor_ripple()         : ir.front();
            diag["lineRmsCurrent"]          = irms.empty() ? pfcInputs.get_last_line_rms_current()        : irms.front();
            diag["inputPower"]              = pin.empty()  ? pfcInputs.get_last_input_power()             : pin.front();
            json perOp = json::array();
            for (size_t i = 0; i < dc.size(); ++i) {
                json row;
                row["operatingPointName"]   = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCyclePeak"]        = dc[i];
                row["peakInductorCurrent"]  = ipk[i];
                row["inductorRipple"]       = ir[i];
                row["lineRmsCurrent"]       = irms[i];
                row["inputPower"]           = pin[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pfcDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

EMSCRIPTEN_KEEPALIVE std::string simulate_pfc_waveforms(std::string pfcInputsString){
    try {
        json pfcInputsJson = json::parse(pfcInputsString);

        OpenMagnetics::PowerFactorCorrection pfcInputs(pfcInputsJson);
        
        // Calculate inductance if not provided
        double inductance;
        if (pfcInputsJson.contains("inductance")) {
            inductance = pfcInputsJson["inductance"];
        } else {
            auto mode = pfcInputs.get_mode().value_or(PfcModes::CONTINUOUS_CONDUCTION_MODE);
            switch (mode) {
                case PfcModes::CONTINUOUS_CONDUCTION_MODE:
                    inductance = pfcInputs.calculate_inductance_ccm(); break;
                case PfcModes::CRITICAL_CONDUCTION_MODE:
                    inductance = pfcInputs.calculate_inductance_crcm(); break;
                case PfcModes::DISCONTINUOUS_CONDUCTION_MODE:
                    inductance = pfcInputs.calculate_inductance_dcm(); break;
                default:
                    throw std::runtime_error("Unsupported PFC mode for simulate_pfc_waveforms");
            }
        }

        // Get design requirements
        auto designRequirements = pfcInputs.process_design_requirements();

        // Get simulation parameters
        double dcResistance = pfcInputsJson.value("dcResistance", 0.1);
        int numberOfCycles = pfcInputsJson.value("numberOfPeriods", 2);
        
        // Set the number of periods for operating point extraction
        pfcInputs.set_num_periods_to_extract(numberOfCycles);
        
        // Simulate and extract waveforms
        auto simWaveforms = pfcInputs.simulate_and_extract_waveforms(
            inductance,
            dcResistance,
            numberOfCycles
        );
        
        // Get operating points from simulation (this will have multiple periods)
        auto operatingPoints = pfcInputs.simulate_and_extract_operating_points(inductance, dcResistance);
        
        // Build result
        json result;
        result["inductance"] = inductance;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        result["converterWaveforms"] = json::array();
        
        // Converter waveforms (for circuit analysis) - Standard format matching other topologies
        json convOp;
        convOp["operatingPointName"] = simWaveforms.operatingPointName;
        convOp["switchingFrequency"] = simWaveforms.switchingFrequency;
        
        // Input voltage (rectified AC) - Standard format with .time and .data
        if (!simWaveforms.inputVoltage.empty()) {
            json inputVoltage;
            inputVoltage["time"] = simWaveforms.time;
            inputVoltage["data"] = simWaveforms.inputVoltage;
            convOp["inputVoltage"] = inputVoltage;
        }
        
        // Input current (inductor current) - Standard format with .time and .data
        if (!simWaveforms.inputCurrent.empty()) {
            json inputCurrent;
            inputCurrent["time"] = simWaveforms.time;
            inputCurrent["data"] = simWaveforms.inputCurrent;
            convOp["inputCurrent"] = inputCurrent;
        }
        
        // Output current (averaged inductor current for PFC) - Standard format
        if (!simWaveforms.outputCurrent.empty()) {
            json outputCurrent;
            outputCurrent["time"] = simWaveforms.time;
            outputCurrent["data"] = simWaveforms.outputCurrent;
            convOp["outputCurrents"] = json::array({outputCurrent});
        }
        
        result["converterWaveforms"].push_back(convOp);
        
        // Add PFC metrics
        result["powerFactor"] = simWaveforms.powerFactor;
        result["efficiency"] = simWaveforms.efficiency;
        result["currentThd"] = simWaveforms.currentThd;

        // Path B diagnostics: see Buck comment for rationale.
        pfcInputs.process();
        {
            json diag;
            diag["computedInductance"]      = pfcInputs.get_computed_inductance();
            diag["actualMode"]              = pfcInputs.get_computed_actual_mode();
            const auto& names = pfcInputs.get_per_op_name();
            const auto& dc    = pfcInputs.get_per_op_duty_cycle_peak();
            const auto& ipk   = pfcInputs.get_per_op_peak_inductor_current();
            const auto& ir    = pfcInputs.get_per_op_inductor_ripple();
            const auto& irms  = pfcInputs.get_per_op_line_rms_current();
            const auto& pin   = pfcInputs.get_per_op_input_power();
            diag["dutyCyclePeak"]           = dc.empty()   ? pfcInputs.get_last_duty_cycle_peak()         : dc.front();
            diag["peakInductorCurrent"]     = ipk.empty()  ? pfcInputs.get_last_peak_inductor_current()   : ipk.front();
            diag["inductorRipple"]          = ir.empty()   ? pfcInputs.get_last_inductor_ripple()         : ir.front();
            diag["lineRmsCurrent"]          = irms.empty() ? pfcInputs.get_last_line_rms_current()        : irms.front();
            diag["inputPower"]              = pin.empty()  ? pfcInputs.get_last_input_power()             : pin.front();
            json perOp = json::array();
            for (size_t i = 0; i < dc.size(); ++i) {
                json row;
                row["operatingPointName"]   = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCyclePeak"]        = dc[i];
                row["peakInductorCurrent"]  = ipk[i];
                row["inductorRipple"]       = ir[i];
                row["lineRmsCurrent"]       = irms[i];
                row["inputPower"]           = pin[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pfcDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

EMSCRIPTEN_KEEPALIVE std::string determine_pfc_mode(std::string pfcInputsString, double inductance){
    try {
        json pfcInputsJson = json::parse(pfcInputsString);
        OpenMagnetics::PowerFactorCorrection pfcInputs(pfcInputsJson);
        
        std::string actualMode = pfcInputs.determine_actual_mode(inductance);
        
        json result;
        result["actualMode"] = actualMode;
        
        // Also return critical inductance for reference
        result["criticalInductance"] = pfcInputs.calculate_inductance_crcm();
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ==========================================
// Common Mode Choke (CMC) Functions
// ==========================================

std::string calculate_cmc_inputs(std::string cmcInputsString){
    try {
        json cmcInputsJson = json::parse(cmcInputsString);

        OpenMagnetics::CommonModeChoke cmcInputs(cmcInputsJson);

        size_t numberOfPeriods = 1;
        if (cmcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = cmcInputsJson["numberOfPeriods"].get<size_t>();
        }
        cmcInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = cmcInputs.process();

        json result;
        to_json(result, inputs);

        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            diag["computedInductance"] = cmcInputs.get_computed_inductance();
            result["cmcDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string calculate_advanced_cmc_inputs(std::string cmcInputsString){
    try {
        json cmcInputsJson = json::parse(cmcInputsString);

        OpenMagnetics::AdvancedCommonModeChoke cmcInputs(cmcInputsJson);

        size_t numberOfPeriods = 1;
        if (cmcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = cmcInputsJson["numberOfPeriods"].get<size_t>();
        }
        cmcInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = cmcInputs.process();

        json result;
        to_json(result, inputs);

        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            diag["computedInductance"] = cmcInputs.get_computed_inductance();
            result["cmcDiagnostics"] = diag;
        }


        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// CMC SPICE Circuit Generation
EMSCRIPTEN_KEEPALIVE std::string generate_cmc_ngspice_circuit(std::string cmcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json cmcInputsJson = json::parse(cmcInputsString);
        
        bool isAdvancedCmc = cmcInputsJson.contains("desiredInductance");
        
        std::unique_ptr<OpenMagnetics::CommonModeChoke> cmcPtr;
        double inductance;
        double frequency = 150000; // Default frequency for CMC
        
        if (isAdvancedCmc) {
            auto advancedCmcPtr = std::make_unique<OpenMagnetics::AdvancedCommonModeChoke>(cmcInputsJson);
            inductance = advancedCmcPtr->get_desired_inductance();
            cmcPtr = std::move(advancedCmcPtr);
        } else {
            cmcPtr = std::make_unique<OpenMagnetics::CommonModeChoke>(cmcInputsJson);
            auto designRequirements = cmcPtr->process_design_requirements();
            
            inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(inductance > 0)) {
                throw std::runtime_error("Unable to calculate CMC inductance");
            }
        }
        
        // CMC generate_ngspice_circuit takes (inductance, frequency) - ignore indices
        std::string netlist = cmcPtr->generate_ngspice_circuit(inductance, frequency);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// CMC LISN Test - runs ngspice with standardized CISPR test circuit
EMSCRIPTEN_KEEPALIVE std::string simulate_cmc_lisn_waveforms(std::string cmcInputsString, double inductance) {
    try {
        json cmcInputsJson = json::parse(cmcInputsString);
        
        OpenMagnetics::CommonModeChoke cmc(cmcInputsJson);
        
        // Get design requirements
        auto designRequirements = cmc.process_design_requirements();
        
        // Get test frequencies from impedance points
        std::vector<double> frequencies;
        if (cmcInputsJson.contains("impedancePoints") && cmcInputsJson["impedancePoints"].is_array()) {
            for (const auto& point : cmcInputsJson["impedancePoints"]) {
                if (point.contains("frequency")) {
                    frequencies.push_back(point["frequency"].get<double>());
                }
            }
        }
        
        // If no frequencies specified, use default
        if (frequencies.empty()) {
            frequencies.push_back(150000); // 150 kHz default
        }
        
        // Run simulation
        auto waveforms = cmc.simulate_and_extract_waveforms(inductance, frequencies);
        
        // Also get operating points for the magnetic data
        auto operatingPoints = cmc.simulate_and_extract_operating_points(inductance);
        
        // Build the result with inputs and converterWaveforms (same format as other wizards)
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: array of CMC test waveforms for visualization
        result["converterWaveforms"] = json::array();
        for (const auto& wf : waveforms) {
            json cwJson;
            cwJson["frequency"] = wf.frequency;
            cwJson["time"] = wf.time;
            cwJson["inputVoltage"] = wf.inputVoltage;
            cwJson["windingCurrents"] = wf.windingCurrents;
            cwJson["lisnVoltage"] = wf.lisnVoltage;
            cwJson["operatingPointName"] = wf.operatingPointName;
            cwJson["commonModeAttenuation"] = wf.commonModeAttenuation;
            cwJson["commonModeImpedance"] = wf.commonModeImpedance;
            cwJson["theoreticalImpedance"] = wf.theoreticalImpedance;
            result["converterWaveforms"].push_back(cwJson);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// CMC Ideal Waveforms - realistic line voltage + switching noise for design
EMSCRIPTEN_KEEPALIVE std::string simulate_cmc_ideal_waveforms(std::string cmcInputsString, double inductance, double parasiticCap_pF, double dvdt_V_ns) {
    try {
        json cmcInputsJson = json::parse(cmcInputsString);

        OpenMagnetics::CommonModeChoke cmc(cmcInputsJson);

        // Get design requirements
        auto designRequirements = cmc.process_design_requirements();

        int numberOfPeriods            = cmcInputsJson.value("numberOfPeriods", 2);
        int numberOfSteadyStatePeriods = cmcInputsJson.value("numberOfSteadyStatePeriods", 10);

        // Run realistic simulation with line + noise
        auto operatingPoints = cmc.simulate_realistic_cmc(
            inductance, parasiticCap_pF, dvdt_V_ns,
            numberOfPeriods, numberOfSteadyStatePeriods);
        
        // Build the result with inputs and converterWaveforms (same format as other wizards)
        json result;
        
        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputs["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputs;
        
        // converterWaveforms: empty array for CMC (realistic simulation doesn't generate converter-style waveforms)
        result["converterWaveforms"] = json::array();

        // Path B diagnostics: see Buck comment for rationale.
        cmc.process();
        {
            json diag;
            diag["computedInductance"] = cmc.get_computed_inductance();
            result["cmcDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// ==========================================
// Differential Mode Choke (DMC) Functions
// ==========================================

std::string calculate_dmc_inputs(std::string dmcInputsString){
    try {
        json dmcInputsJson = json::parse(dmcInputsString);

        OpenMagnetics::DifferentialModeChoke dmcInputs(dmcInputsJson);
        auto inputs = dmcInputs.process();

        json result;
        to_json(result, inputs);

        // Honor numberOfPeriods so the frontend doesn't have to tile the
        // waveforms in JS. DMC's process() always emits a single period;
        // we replicate it here using the shared helper, matching the
        // pattern used by every other converter binding above.
        size_t numberOfPeriods = 1;
        if (dmcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = dmcInputsJson["numberOfPeriods"].get<size_t>();
        }
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        {
            json diag;
            diag["computedInductance"]      = dmcInputs.get_computed_inductance();
            diag["computedMinFrequency"]    = dmcInputs.get_computed_min_frequency();
            diag["computedMaxFrequency"]    = dmcInputs.get_computed_max_frequency();
            diag["impedanceAtMinFrequency"] = dmcInputs.get_computed_impedance_at_min_freq();
            diag["numberWindings"]          = dmcInputs.get_computed_number_windings();
            result["dmcDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string verify_dmc_attenuation(std::string dmcInputsString, double inductance, double capacitance) {
    try {
        json dmcInputsJson = json::parse(dmcInputsString);
        OpenMagnetics::DifferentialModeChoke dmc(dmcInputsJson);

        std::optional<double> cap = (capacitance > 0) ? std::optional<double>(capacitance) : std::nullopt;
        auto results = dmc.verify_attenuation(inductance, cap);

        json result = json::array();
        for (const auto& r : results) {
            json rJson;
            rJson["frequency"] = r.frequency;
            rJson["requiredAttenuation"] = r.requiredAttenuation;
            rJson["measuredAttenuation"] = r.measuredAttenuation;
            rJson["theoreticalAttenuation"] = r.theoreticalAttenuation;
            rJson["passed"] = r.passed;
            rJson["message"] = r.message;
            result.push_back(rJson);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string propose_dmc_design(std::string dmcInputsString) {
    try {
        json dmcInputsJson = json::parse(dmcInputsString);
        OpenMagnetics::DifferentialModeChoke dmc(dmcInputsJson);

        auto proposal = dmc.propose_design();
        return proposal.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_dmc_waveforms(std::string dmcInputsString, double inductance) {
    try {
        json dmcInputsJson = json::parse(dmcInputsString);
        OpenMagnetics::DifferentialModeChoke dmc(dmcInputsJson);

        // Get test frequencies from minimum impedance requirements
        std::vector<double> frequencies;
        auto minimumImpedance = dmc.get_minimum_impedance();
        if (minimumImpedance) {
            for (const auto& imp : *minimumImpedance) {
                frequencies.push_back(imp.get_frequency());
            }
        }
        
        // If no impedance points specified, use default EMI test frequencies
        if (frequencies.empty()) {
            frequencies = {150000, 500000, 1000000, 10000000, 30000000};
        }

        // Honor numberOfPeriods (matches the pattern used by every other
        // converter binding: pass through to the underlying simulator,
        // which sets ngspice's extractOnePeriod + numberOfPeriods).
        size_t numberOfPeriods = 1;
        if (dmcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = dmcInputsJson["numberOfPeriods"].get<size_t>();
        }

        auto waveforms = dmc.simulate_and_extract_waveforms(inductance, frequencies, numberOfPeriods);

        json result = json::array();
        for (const auto& wf : waveforms) {
            json wfJson;
            wfJson["time"] = wf.time;
            wfJson["frequency"] = wf.frequency;
            wfJson["inputVoltage"] = wf.inputVoltage;
            wfJson["outputVoltage"] = wf.outputVoltage;
            wfJson["inductorCurrent"] = wf.inductorCurrent;
            wfJson["operatingPointName"] = wf.operatingPointName;
            wfJson["dmAttenuation"] = wf.dmAttenuation;
            result.push_back(wfJson);
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

// DMC SPICE Circuit Generation
EMSCRIPTEN_KEEPALIVE std::string generate_dmc_ngspice_circuit(std::string dmcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex){
    try {
        json dmcInputsJson = json::parse(dmcInputsString);

        OpenMagnetics::DifferentialModeChoke dmc(dmcInputsJson);

        double inductance;
        double frequency = 150000; // Default test frequency for DMC

        // Prefer an explicit minimumInductance from the wizard params; fall
        // back to a proposed inductance derived from the attenuation /
        // impedance requirements via propose_design().
        if (dmcInputsJson.contains("minimumInductance") && dmcInputsJson["minimumInductance"].is_number()) {
            inductance = dmcInputsJson["minimumInductance"].get<double>();
        } else {
            auto proposal = dmc.propose_design();
            if (proposal.contains("inductance") && proposal["inductance"].is_number()) {
                inductance = proposal["inductance"].get<double>();
            } else if (proposal.contains("minimumInductance") && proposal["minimumInductance"].is_number()) {
                inductance = proposal["minimumInductance"].get<double>();
            } else {
                throw std::runtime_error("Unable to determine DMC inductance for SPICE generation");
            }
        }

        std::string netlist = dmc.generate_ngspice_circuit(inductance, frequency);
        return netlist;
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

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

        json result;
        to_json(result, coefficientsPerRange);
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

std::string plot_turns(std::string magneticString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        
        auto magneticJson = json::parse(magneticString);
        
        OpenMagnetics::Magnetic magnetic(magneticJson);

        // Ensure the coil is wound; otherwise paint_coil_turns throws COIL_NOT_PROCESSED.
        // Catalog magnetics often arrive with only functionalDescription populated.
        {
            auto coil = magnetic.get_mutable_coil();
            if (!coil.get_turns_description() || coil.get_turns_description()->empty()) {
                coil.wind();
                magnetic.set_coil(coil);
            }
        }

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_core(magnetic);
        painter.paint_bobbin(magnetic);
        painter.paint_coil_turns(magnetic);
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


std::string plot_magnetic_field(std::string magneticString, std::string operatingPointString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        // For toroidal cores, ensure the coil is wound to generate additional_coordinates
        auto coil = magnetic.get_mutable_coil();
        auto core = magnetic.get_mutable_core();
        if (core.get_shape_family() == OpenMagnetics::CoreShapeFamily::T) {
            if (!coil.get_turns_description() || coil.get_turns_description()->empty()) {
                coil.wind();
                magnetic.set_coil(coil);
            }
        }

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_magnetic_field(operatingPoint, magnetic);
        painter.paint_core(magnetic);
        // painter.paint_bobbin(magnetic);
        // Paint turns for H field, skip insulation tape and margin for cleaner visualization
        painter.paint_coil_turns(magnetic, true);
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


std::string plot_electric_field(std::string magneticString, std::string operatingPointString) {
    try {
        OpenMagnetics::Settings::GetInstance().set_painter_simple_litz(true);
        OpenMagnetics::Settings::GetInstance().set_painter_advanced_litz(false);
        std::filesystem::path emptyFilepath;
        OpenMagnetics::Magnetic magnetic(json::parse(magneticString));
        OperatingPoint operatingPoint(json::parse(operatingPointString));
        
        // For toroidal cores, ensure the coil is wound to generate additional_coordinates
        auto coil = magnetic.get_mutable_coil();
        auto core = magnetic.get_mutable_core();
        if (core.get_shape_family() == OpenMagnetics::CoreShapeFamily::T) {
            if (!coil.get_turns_description() || coil.get_turns_description()->empty()) {
                coil.wind();
                magnetic.set_coil(coil);
            }
        }

        OpenMagnetics::Painter painter(emptyFilepath);
        painter.paint_electric_field(operatingPoint, magnetic);
        painter.paint_core(magnetic);
        // painter.paint_bobbin(magnetic);
        // Paint turns for E field, skip insulation tape and margin for cleaner visualization
        painter.paint_coil_turns(magnetic, true);
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
        
        // For toroidal cores, ensure the coil is wound
        auto coil = magnetic.get_mutable_coil();
        auto core = magnetic.get_mutable_core();
        if (core.get_shape_family() == OpenMagnetics::CoreShapeFamily::T) {
            if (!coil.get_turns_description() || coil.get_turns_description()->empty()) {
                coil.wind();
                magnetic.set_coil(coil);
            }
        }
        
        // Get ambient temperature from operating point
        double ambientTemperature = operatingPoint.get_conditions().get_ambient_temperature();
        
        // Run magnetic simulation to get losses
        OpenMagnetics::MagneticSimulator magneticSimulator;
        OpenMagnetics::Mas mas;
        mas.set_magnetic(magnetic);
        mas.get_mutable_inputs().set_operating_points({operatingPoint});  // Set the operating point for simulation
        auto simulatedMas = magneticSimulator.simulate(mas);
        
        double coreLosses = 0.0;
        double windingLosses = 0.0;
        std::optional<OpenMagnetics::WindingLossesOutput> windingLossesOutput;
        
        if (!simulatedMas.get_outputs().empty()) {
            auto outputs = simulatedMas.get_outputs()[0];
            if (outputs.get_core_losses().has_value()) {
                coreLosses = outputs.get_core_losses().value().get_core_losses();
            }
            if (outputs.get_winding_losses().has_value()) {
                windingLosses = outputs.get_winding_losses().value().get_winding_losses();
                // Also get the detailed per-turn losses (required for toroidal cores)
                windingLossesOutput = outputs.get_winding_losses().value();
            }
        }
        
        // Create temperature configuration
        OpenMagnetics::TemperatureConfig config;
        config.ambientTemperature = ambientTemperature;
        config.coreLosses = coreLosses;
        config.windingLosses = windingLosses;
        // Set per-turn losses (required for toroidal core thermal analysis)
        if (windingLossesOutput) {
            config.windingLossesOutput = windingLossesOutput;
        }
        
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
        auto [areaFillingFactor, aux] = coil.calculate_filling_factor();
        auto [overlappingFillingFactor, contiguousFillingFactor] = aux;
        json result;
        result["areaFillingFactor"] = areaFillingFactor;
        result["overlappingFillingFactor"] = overlappingFillingFactor;
        result["contiguousFillingFactor"] = contiguousFillingFactor;
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
        settingsJson["coilIncludeAdditionalCoordinates"] = OpenMagnetics::Settings::GetInstance().get_coil_include_additional_coordinates();

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
    if (settingsJson.contains("coilIncludeAdditionalCoordinates")) {
        OpenMagnetics::Settings::GetInstance().set_coil_include_additional_coordinates(settingsJson["coilIncludeAdditionalCoordinates"]);
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
std::string calculate_llc_inputs(std::string llcInputsString);
std::string simulate_llc_ideal_waveforms(std::string llcInputsString);
std::string simulate_dab_ideal_waveforms(std::string dabInputsString);
std::string calculate_cllc_inputs(std::string cllcInputsString);
std::string simulate_cllc_ideal_waveforms(std::string cllcInputsString);
std::string calculate_dab_inputs(std::string dabInputsString);
std::string calculate_psfb_inputs(std::string psfbInputsString);
std::string simulate_psfb_ideal_waveforms(std::string psfbInputsString);
std::string calculate_pshb_inputs(std::string pshbInputsString);
std::string simulate_pshb_ideal_waveforms(std::string pshbInputsString);
std::string calculate_ahb_inputs(std::string ahbInputsString);
std::string simulate_ahb_ideal_waveforms(std::string ahbInputsString);
std::string calculate_clllc_inputs(std::string clllcInputsString);
std::string calculate_advanced_clllc_inputs(std::string clllcInputsString);
std::string simulate_clllc_ideal_waveforms(std::string clllcInputsString);
std::string calculate_cuk_inputs(std::string cukInputsString);
std::string calculate_advanced_cuk_inputs(std::string cukInputsString);
std::string simulate_cuk_ideal_waveforms(std::string cukInputsString);
std::string calculate_four_switch_buck_boost_inputs(std::string fsbbInputsString);
std::string calculate_advanced_four_switch_buck_boost_inputs(std::string fsbbInputsString);
std::string simulate_four_switch_buck_boost_ideal_waveforms(std::string fsbbInputsString);
std::string calculate_weinberg_inputs(std::string weinbergInputsString);
std::string calculate_advanced_weinberg_inputs(std::string weinbergInputsString);
std::string simulate_weinberg_ideal_waveforms(std::string weinbergInputsString);
std::string calculate_zeta_inputs(std::string zetaInputsString);
std::string calculate_advanced_zeta_inputs(std::string zetaInputsString);
std::string simulate_zeta_ideal_waveforms(std::string zetaInputsString);
std::string calculate_src_inputs(std::string srcInputsString);
std::string simulate_src_ideal_waveforms(std::string srcInputsString);
std::string calculate_vienna_inputs(std::string viennaInputsString);
std::string simulate_vienna_ideal_waveforms(std::string viennaInputsString);
std::string process_current_transformer(std::string ctInputsString);

// SPICE Code Generation forward declarations
EMSCRIPTEN_KEEPALIVE std::string generate_flyback_ngspice_circuit(std::string flybackInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_buck_ngspice_circuit(std::string buckInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_boost_ngspice_circuit(std::string boostInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_sepic_ngspice_circuit(std::string sepicInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_push_pull_ngspice_circuit(std::string pushPullInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_forward_ngspice_circuit(std::string forwardInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_two_switch_forward_ngspice_circuit(std::string forwardInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_active_clamp_forward_ngspice_circuit(std::string forwardInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_isolated_buck_ngspice_circuit(std::string isolatedBuckInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_isolated_buck_boost_ngspice_circuit(std::string isolatedBuckBoostInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_llc_ngspice_circuit(std::string llcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_cllc_ngspice_circuit(std::string cllcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_src_ngspice_circuit(std::string srcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_dab_ngspice_circuit(std::string dabInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_psfb_ngspice_circuit(std::string psfbInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string generate_cmc_ngspice_circuit(std::string cmcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);
EMSCRIPTEN_KEEPALIVE std::string simulate_cmc_lisn_waveforms(std::string cmcInputsString, double inductance);
EMSCRIPTEN_KEEPALIVE std::string simulate_cmc_ideal_waveforms(std::string cmcInputsString, double inductance, double parasiticCap_pF, double dvdt_V_ns);
EMSCRIPTEN_KEEPALIVE std::string generate_dmc_ngspice_circuit(std::string dmcInputsString, size_t inputVoltageIndex, size_t operatingPointIndex);

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
                                                  std::string constraintsString,
                                                  bool useContext) {
    try {
        OpenMagnetics::Inputs inputs(json::parse(inputsString));
        std::map<std::string, double> weightsKeysString = json::parse(weightsString);
        std::map<OpenMagnetics::CoreAdviser::CoreAdviserFilters, double> weights;
        double externalSum = 0;
        for (auto const& pair : weightsKeysString) externalSum += pair.second;
        for (auto const& [name, weight] : weightsKeysString) {
            OpenMagnetics::CoreAdviser::CoreAdviserFilters filter;
            OpenMagnetics::from_json(name, filter);
            weights[filter] = externalSum > 0 ? weight / externalSum : weight;
        }
        auto constraints = parse_constraints_json(constraintsString);
        OpenMagnetics::CoreAdviser adviser;
        const OpenMagnetics::LibraryContext* ctx = useContext ? &g_libraryContext : nullptr;
        auto results = adviser.get_advised_core(inputs, weights,
                                                 (size_t)maximumNumberResults,
                                                 ctx, constraints);
        json out = json::array();
        for (auto& [mas, score] : results) {
            json entry;
            to_json(entry["mas"], mas);
            entry["score"] = score;
            out.push_back(std::move(entry));
        }
        return out.dump();
    } catch (const std::exception& e) {
        json err; err["error"] = e.what();
        return err.dump();
    }
}

std::string calculate_advised_magnetics_with_context(std::string inputsString,
                                                      int maximumNumberResults,
                                                      std::string constraintsString,
                                                      bool useContext) {
    try {
        OpenMagnetics::Inputs inputs(json::parse(inputsString));
        auto constraints = parse_constraints_json(constraintsString);
        OpenMagnetics::MagneticAdviser adviser;
        const OpenMagnetics::LibraryContext* ctx = useContext ? &g_libraryContext : nullptr;
        auto results = adviser.get_advised_magnetic(inputs,
                                                     (size_t)maximumNumberResults,
                                                     ctx, constraints);
        json out = json::array();
        for (auto& [mas, score] : results) {
            json entry;
            to_json(entry["mas"], mas);
            entry["score"] = score;
            out.push_back(std::move(entry));
        }
        return out.dump();
    } catch (const std::exception& e) {
        json err; err["error"] = e.what();
        return err.dump();
    }
}

EMSCRIPTEN_BINDINGS(my_bindings) {
    function("get_constants", &get_constants);
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
    function("get_available_core_manufacturers", &get_available_core_manufacturers);
    function("get_available_core_shape_families", &get_available_core_shape_families);
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
    function("wind_planar", &wind_planar);
    function("wind_by_sections", &wind_by_sections);
    function("wind_by_layers", &wind_by_layers);
    function("wind_by_turns", &wind_by_turns);
    function("delimit_and_compact", &delimit_and_compact);
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
    function("calculate_capacitance_models_between_windings", &calculate_capacitance_models_between_windings);
    function("get_available_core_losses_methods", &get_available_core_losses_methods);
    function("get_all_magnetic_field_strength_models", &get_all_magnetic_field_strength_models);
    function("get_all_fringing_effect_models", &get_all_fringing_effect_models);
    function("get_all_reluctance_models", &get_all_reluctance_models);
    function("get_all_winding_skin_effect_models", &get_all_winding_skin_effect_models);
    function("get_all_winding_proximity_effect_models", &get_all_winding_proximity_effect_models);
    function("get_all_stray_capacitance_models", &get_all_stray_capacitance_models);
    function("calculate_resistance_matrix", &calculate_resistance_matrix);
    function("calculate_flyback_inputs", &calculate_flyback_inputs);
    function("calculate_advanced_flyback_inputs", &calculate_advanced_flyback_inputs);
    function("simulate_flyback_ideal_waveforms", &simulate_flyback_ideal_waveforms);
    function("simulate_flyback_with_magnetic", &simulate_flyback_with_magnetic);
    function("calculate_isolated_buck_inputs", &calculate_isolated_buck_inputs);
    function("calculate_advanced_isolated_buck_inputs", &calculate_advanced_isolated_buck_inputs);
    function("calculate_isolated_buck_boost_inputs", &calculate_isolated_buck_boost_inputs);
    function("calculate_advanced_isolated_buck_boost_inputs", &calculate_advanced_isolated_buck_boost_inputs);
    function("simulate_isolated_buck_boost_ideal_waveforms", &simulate_isolated_buck_boost_ideal_waveforms);
    function("simulate_isolated_buck_ideal_waveforms", &simulate_isolated_buck_ideal_waveforms);
    function("calculate_clllc_inputs", &calculate_clllc_inputs);
    function("calculate_advanced_clllc_inputs", &calculate_advanced_clllc_inputs);
    function("simulate_clllc_ideal_waveforms", &simulate_clllc_ideal_waveforms);
    function("calculate_cuk_inputs", &calculate_cuk_inputs);
    function("calculate_advanced_cuk_inputs", &calculate_advanced_cuk_inputs);
    function("simulate_cuk_ideal_waveforms", &simulate_cuk_ideal_waveforms);
    function("calculate_four_switch_buck_boost_inputs", &calculate_four_switch_buck_boost_inputs);
    function("calculate_advanced_four_switch_buck_boost_inputs", &calculate_advanced_four_switch_buck_boost_inputs);
    function("simulate_four_switch_buck_boost_ideal_waveforms", &simulate_four_switch_buck_boost_ideal_waveforms);
    function("calculate_weinberg_inputs", &calculate_weinberg_inputs);
    function("calculate_advanced_weinberg_inputs", &calculate_advanced_weinberg_inputs);
    function("simulate_weinberg_ideal_waveforms", &simulate_weinberg_ideal_waveforms);
    function("calculate_zeta_inputs", &calculate_zeta_inputs);
    function("calculate_advanced_zeta_inputs", &calculate_advanced_zeta_inputs);
    function("simulate_zeta_ideal_waveforms", &simulate_zeta_ideal_waveforms);
    function("calculate_src_inputs", &calculate_src_inputs);
    function("simulate_src_ideal_waveforms", &simulate_src_ideal_waveforms);
    function("calculate_vienna_inputs", &calculate_vienna_inputs);
    function("simulate_vienna_ideal_waveforms", &simulate_vienna_ideal_waveforms);
    function("process_current_transformer", &process_current_transformer);
    function("calculate_buck_inputs", &calculate_buck_inputs);
    function("calculate_advanced_buck_inputs", &calculate_advanced_buck_inputs);
    function("simulate_buck_ideal_waveforms", &simulate_buck_ideal_waveforms);
    function("calculate_boost_inputs", &calculate_boost_inputs);
    function("calculate_advanced_boost_inputs", &calculate_advanced_boost_inputs);
    function("simulate_boost_ideal_waveforms", &simulate_boost_ideal_waveforms);
    function("calculate_sepic_inputs", &calculate_sepic_inputs);
    function("calculate_advanced_sepic_inputs", &calculate_advanced_sepic_inputs);
    function("simulate_sepic_ideal_waveforms", &simulate_sepic_ideal_waveforms);
    function("calculate_push_pull_inputs", &calculate_push_pull_inputs);
    function("calculate_advanced_push_pull_inputs", &calculate_advanced_push_pull_inputs);
    function("simulate_push_pull_ideal_waveforms", &simulate_push_pull_ideal_waveforms);
    // Forward converter functions
    function("calculate_single_switch_forward_inputs", &calculate_single_switch_forward_inputs);
    function("calculate_advanced_single_switch_forward_inputs", &calculate_advanced_single_switch_forward_inputs);
    function("simulate_forward_ideal_waveforms", &simulate_forward_ideal_waveforms);
    function("calculate_active_clamp_forward_inputs", &calculate_active_clamp_forward_inputs);
    function("calculate_advanced_active_clamp_forward_inputs", &calculate_advanced_active_clamp_forward_inputs);
    function("simulate_active_clamp_forward_ideal_waveforms", &simulate_active_clamp_forward_ideal_waveforms);
    function("calculate_two_switch_forward_inputs", &calculate_two_switch_forward_inputs);
    function("calculate_advanced_two_switch_forward_inputs", &calculate_advanced_two_switch_forward_inputs);
    function("simulate_two_switch_forward_ideal_waveforms", &simulate_two_switch_forward_ideal_waveforms);
    
    // SPICE Code Generation functions
    function("generate_flyback_ngspice_circuit", &generate_flyback_ngspice_circuit);
    function("generate_buck_ngspice_circuit", &generate_buck_ngspice_circuit);
    function("generate_boost_ngspice_circuit", &generate_boost_ngspice_circuit);
    function("generate_sepic_ngspice_circuit", &generate_sepic_ngspice_circuit);
    function("generate_push_pull_ngspice_circuit", &generate_push_pull_ngspice_circuit);
    function("generate_forward_ngspice_circuit", &generate_forward_ngspice_circuit);
    function("generate_two_switch_forward_ngspice_circuit", &generate_two_switch_forward_ngspice_circuit);
    function("generate_active_clamp_forward_ngspice_circuit", &generate_active_clamp_forward_ngspice_circuit);
    function("generate_isolated_buck_ngspice_circuit", &generate_isolated_buck_ngspice_circuit);
    function("generate_isolated_buck_boost_ngspice_circuit", &generate_isolated_buck_boost_ngspice_circuit);
    function("generate_llc_ngspice_circuit", &generate_llc_ngspice_circuit);
    function("generate_cllc_ngspice_circuit", &generate_cllc_ngspice_circuit);
    function("generate_src_ngspice_circuit", &generate_src_ngspice_circuit);
    function("generate_dab_ngspice_circuit", &generate_dab_ngspice_circuit);
    function("generate_psfb_ngspice_circuit", &generate_psfb_ngspice_circuit);
    function("generate_cmc_ngspice_circuit", &generate_cmc_ngspice_circuit);
    function("generate_dmc_ngspice_circuit", &generate_dmc_ngspice_circuit);
    function("generate_cuk_ngspice_circuit", &generate_cuk_ngspice_circuit);
    function("generate_zeta_ngspice_circuit", &generate_zeta_ngspice_circuit);
    function("generate_four_switch_buck_boost_ngspice_circuit", &generate_four_switch_buck_boost_ngspice_circuit);
    function("generate_weinberg_ngspice_circuit", &generate_weinberg_ngspice_circuit);
    function("generate_clllc_ngspice_circuit", &generate_clllc_ngspice_circuit);
    function("generate_pshb_ngspice_circuit", &generate_pshb_ngspice_circuit);
    function("generate_ahb_ngspice_circuit", &generate_ahb_ngspice_circuit);

    function("calculate_pfc_inputs", &calculate_pfc_inputs);
    function("simulate_pfc_waveforms", &simulate_pfc_waveforms);
    function("determine_pfc_mode", &determine_pfc_mode);
    function("calculate_cmc_inputs", &calculate_cmc_inputs);
    function("calculate_advanced_cmc_inputs", &calculate_advanced_cmc_inputs);
    function("simulate_cmc_lisn_waveforms", &simulate_cmc_lisn_waveforms);
    function("simulate_cmc_ideal_waveforms", &simulate_cmc_ideal_waveforms);
    function("calculate_dmc_inputs", &calculate_dmc_inputs);
    function("verify_dmc_attenuation", &verify_dmc_attenuation);
    function("propose_dmc_design", &propose_dmc_design);
    function("simulate_dmc_waveforms", &simulate_dmc_waveforms);
    function("get_only_temperature_dependent_indexes", &get_only_temperature_dependent_indexes);
    function("get_only_frequency_dependent_indexes", &get_only_frequency_dependent_indexes);
    function("get_only_magnetic_field_dc_bias_dependent_indexes", &get_only_magnetic_field_dc_bias_dependent_indexes);
    function("create_simple_bobbin_from_core", &create_simple_bobbin_from_core);
    function("create_simple_bobbin_from_core_with_custom_thickness", &create_simple_bobbin_from_core_with_custom_thickness);
    function("create_simple_bobbin_from_core_with_custom_thicknesses", &create_simple_bobbin_from_core_with_custom_thicknesses);
    function("mas_autocomplete", &mas_autocomplete);
    function("calculate_steinmetz_coefficients", &calculate_steinmetz_coefficients);
    function("get_initial_permeability_equations", &get_initial_permeability_equations);
    function("get_core_volumetric_losses_equations", &get_core_volumetric_losses_equations);
    function("calculate_complex_permeability", &calculate_complex_permeability);
    function("plot_core", &plot_core);
    function("plot_sections", &plot_sections);
    function("plot_layers", &plot_layers);
    function("plot_turns", &plot_turns);
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
    function("calculate_llc_inputs", &calculate_llc_inputs);
    function("simulate_llc_ideal_waveforms", &simulate_llc_ideal_waveforms);
    function("simulate_dab_ideal_waveforms", &simulate_dab_ideal_waveforms);
    function("calculate_cllc_inputs", &calculate_cllc_inputs);
    function("simulate_cllc_ideal_waveforms", &simulate_cllc_ideal_waveforms);
    function("calculate_dab_inputs", &calculate_dab_inputs);
    function("calculate_psfb_inputs", &calculate_psfb_inputs);
    function("simulate_psfb_ideal_waveforms", &simulate_psfb_ideal_waveforms);
    function("calculate_pshb_inputs", &calculate_pshb_inputs);
    function("simulate_pshb_ideal_waveforms", &simulate_pshb_ideal_waveforms);
    function("calculate_ahb_inputs", &calculate_ahb_inputs);
    function("simulate_ahb_ideal_waveforms", &simulate_ahb_ideal_waveforms);
    
    // New integrated converter processing functions
    function("process_converter", &process_converter);
    function("design_magnetics_from_converter", &design_magnetics_from_converter);
    
    register_map<std::string, double>("map<string, double>");
    register_map<std::string, std::string>("map<string, string>");
    // register_map<std::string, std::map<std::string, std::string>>("map<string, map<string, string>>");
    register_vector<std::string>("vector<std::string>");
    register_vector<int>("vector<int>");
    register_vector<double>("vector<double>");
    register_vector<size_t>("vector<size_t>");

};

// New topology functions - DAB, LLC, CLLC, PSFB

std::string calculate_dab_inputs(std::string dabInputsString) {
    try {
        json dabInputsJson = json::parse(dabInputsString);

        OpenMagnetics::AdvancedDab dabInputs(dabInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (dabInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = dabInputsJson["numberOfPeriods"].get<size_t>();
        }
        dabInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto inputs = dabInputs.process();
        
        json result;
        to_json(result, inputs);
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        // Append per-op diagnostics from the Dab object (populated after process())
        {
            json dabDiag;
            const auto& names = dabInputs.get_per_op_name();
            const auto& mt    = dabInputs.get_per_op_modulation_type();
            const auto& zp    = dabInputs.get_per_op_zvs_margin_primary();
            const auto& zs    = dabInputs.get_per_op_zvs_margin_secondary();
            const auto& d3    = dabInputs.get_per_op_d3_rad();
            const auto& vr    = dabInputs.get_per_op_voltage_conversion_ratio();
            dabDiag["modulationType"]           = mt.empty() ? dabInputs.get_last_modulation_type()           : mt.front();
            dabDiag["computedD3Deg"]            = (d3.empty() ? dabInputs.get_last_d3_rad()                   : d3.front()) * 180.0 / M_PI;
            dabDiag["zvsMarginPrimaryDeg"]      = (zp.empty() ? dabInputs.get_last_zvs_margin_primary()       : zp.front()) * 180.0 / M_PI;
            dabDiag["zvsMarginSecondaryDeg"]    = (zs.empty() ? dabInputs.get_last_zvs_margin_secondary()     : zs.front()) * 180.0 / M_PI;
            dabDiag["computedSeriesInductance"] = dabInputs.get_computed_series_inductance();
            dabDiag["voltageConversionRatio"]   = vr.empty() ? dabInputs.get_last_voltage_conversion_ratio() : vr.front();
            json perOp = json::array();
            for (size_t i = 0; i < mt.size(); ++i) {
                json row;
                row["operatingPointName"]        = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["modulationType"]            = mt[i];
                row["zvsMarginPrimaryDeg"]       = zp[i] * 180.0 / M_PI;
                row["zvsMarginSecondaryDeg"]     = zs[i] * 180.0 / M_PI;
                row["computedD3Deg"]             = d3[i] * 180.0 / M_PI;
                row["voltageConversionRatio"]    = vr[i];
                perOp.push_back(row);
            }
            dabDiag["perOp"] = perOp;
            result["dabDiagnostics"] = dabDiag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

EMSCRIPTEN_KEEPALIVE std::string calculate_llc_inputs(std::string llcInputsString){
    try {
        json llcInputsJson = json::parse(llcInputsString);

        OpenMagnetics::Llc llcInputs(llcInputsJson);

        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (llcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = llcInputsJson["numberOfPeriods"].get<size_t>();
        }
        llcInputs.set_num_periods_to_extract(numberOfPeriods);

        // Optional user overrides for the resonant tank values. The Llc class
        // honours these via set_user_resonant_{inductance,capacitance}, but the
        // JSON constructor doesn't auto-bind them — wire them here.
        if (llcInputsJson.contains("desiredResonantInductance") &&
            llcInputsJson["desiredResonantInductance"].is_number()) {
            llcInputs.set_user_resonant_inductance(
                llcInputsJson["desiredResonantInductance"].get<double>());
        }
        if (llcInputsJson.contains("desiredResonantCapacitance") &&
            llcInputsJson["desiredResonantCapacitance"].is_number()) {
            llcInputs.set_user_resonant_capacitance(
                llcInputsJson["desiredResonantCapacitance"].get<double>());
        }

        auto inputs = llcInputs.process();

        json result;
        to_json(result, inputs);

        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        // Expose the Nielsen TDA diagnostics + computed tank values so the
        // frontend can show them without re-running the solver.
        json diag;
        diag["computedResonantInductance"]  = llcInputs.get_computed_resonant_inductance();
        diag["computedResonantCapacitance"] = llcInputs.get_computed_resonant_capacitance();
        diag["computedInductanceRatio"]     = llcInputs.get_computed_inductance_ratio();
        diag["lipFrequency"]                = llcInputs.get_lip_frequency();
        diag["lipInputVoltage"]             = llcInputs.get_lip_input_voltage();
        diag["lastSubStateSequence"]        = llcInputs.get_last_sub_state_sequence();
        {
            const auto& names = llcInputs.get_per_op_name();
            const auto& mode  = llcInputs.get_per_op_mode();
            const auto& res   = llcInputs.get_per_op_steady_state_residual();
            const auto& zvs   = llcInputs.get_per_op_zvs_margin_lagging();
            const auto& ipk   = llcInputs.get_per_op_primary_peak_current();
            diag["lastMode"]                = mode.empty() ? llcInputs.get_last_mode()                  : mode.front();
            diag["lastSteadyStateResidual"] = res.empty()  ? llcInputs.get_last_steady_state_residual() : res.front();
            diag["lastZvsMarginLagging"]    = zvs.empty()  ? llcInputs.get_last_zvs_margin_lagging()    : zvs.front();
            diag["lastPrimaryPeakCurrent"]  = ipk.empty()  ? llcInputs.get_last_primary_peak_current()  : ipk.front();
            json perOp = json::array();
            for (size_t i = 0; i < mode.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["lastMode"]              = mode[i];
                row["steadyStateResidual"]   = res[i];
                row["zvsMarginLagging"]      = zvs[i];
                row["primaryPeakCurrent"]    = ipk[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
        }
        result["llcDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_llc_ideal_waveforms(std::string llcInputsString) {
    try {
        json llcInputsJson = json::parse(llcInputsString);

        OpenMagnetics::Llc llcInputs(llcInputsJson);

        // Optional user overrides for the resonant tank values.
        if (llcInputsJson.contains("desiredResonantInductance") &&
            llcInputsJson["desiredResonantInductance"].is_number()) {
            llcInputs.set_user_resonant_inductance(
                llcInputsJson["desiredResonantInductance"].get<double>());
        }
        if (llcInputsJson.contains("desiredResonantCapacitance") &&
            llcInputsJson["desiredResonantCapacitance"].is_number()) {
            llcInputs.set_user_resonant_capacitance(
                llcInputsJson["desiredResonantCapacitance"].get<double>());
        }

        auto designRequirements = llcInputs.process_design_requirements();
        double magnetizingInductance = llcInputsJson.value("magnetizingInductance", 200e-6);

        // Get turns ratios from design requirements
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif

        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        // Read number of periods from input (default to 2)
        size_t numberOfPeriods = 2;
        if (llcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = llcInputsJson["numberOfPeriods"].get<size_t>();
        }
        
        // Read number of steady state periods from input (default to 3)
        size_t numberOfSteadyStatePeriods = 3;
        if (llcInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = llcInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        
        // Set the number of periods for the LLC simulation
        llcInputs.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        llcInputs.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));
        
        // Run both simulations (topology waveforms + operating points)
        auto topologyWaveforms = llcInputs.simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        auto operatingPoints = llcInputs.simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance, numberOfPeriods);

        // Build the result with inputs and converterWaveforms
        json result;

        // inputs: OpenMagnetics::Inputs containing designRequirements and operatingPoints
        json inputsJson;
        inputsJson["designRequirements"] = json();
        to_json(inputsJson["designRequirements"], designRequirements);
        inputsJson["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJson["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJson;

        // converterWaveforms: array of OpenMagnetics::ConverterWaveforms
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // Nielsen TDA diagnostics + computed tank values.
        json diag;
        diag["computedResonantInductance"]  = llcInputs.get_computed_resonant_inductance();
        diag["computedResonantCapacitance"] = llcInputs.get_computed_resonant_capacitance();
        diag["computedInductanceRatio"]     = llcInputs.get_computed_inductance_ratio();
        diag["lipFrequency"]                = llcInputs.get_lip_frequency();
        diag["lipInputVoltage"]             = llcInputs.get_lip_input_voltage();
        diag["lastSubStateSequence"]        = llcInputs.get_last_sub_state_sequence();
        {
            const auto& names = llcInputs.get_per_op_name();
            const auto& mode  = llcInputs.get_per_op_mode();
            const auto& res   = llcInputs.get_per_op_steady_state_residual();
            const auto& zvs   = llcInputs.get_per_op_zvs_margin_lagging();
            const auto& ipk   = llcInputs.get_per_op_primary_peak_current();
            diag["lastMode"]                = mode.empty() ? llcInputs.get_last_mode()                  : mode.front();
            diag["lastSteadyStateResidual"] = res.empty()  ? llcInputs.get_last_steady_state_residual() : res.front();
            diag["lastZvsMarginLagging"]    = zvs.empty()  ? llcInputs.get_last_zvs_margin_lagging()    : zvs.front();
            diag["lastPrimaryPeakCurrent"]  = ipk.empty()  ? llcInputs.get_last_primary_peak_current()  : ipk.front();
            json perOp = json::array();
            for (size_t i = 0; i < mode.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["lastMode"]              = mode[i];
                row["steadyStateResidual"]   = res[i];
                row["zvsMarginLagging"]      = zvs[i];
                row["primaryPeakCurrent"]    = ipk[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
        }
        result["llcDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_dab_ideal_waveforms(std::string dabInputsString) {
    try {
        json dabInputsJson = json::parse(dabInputsString);

        // Use AdvancedDab so that desiredTurnsRatios and desiredMagnetizingInductance
        // from the UI are respected. Using base Dab recomputes N=Vin/Vout which can
        // produce d=1 (degenerate) when the user specifies a non-default turns ratio,
        // causing ngspice to hang indefinitely.
        OpenMagnetics::AdvancedDab dabInputs(dabInputsJson);

        auto t0 = std::chrono::steady_clock::now();
        auto inputs = dabInputs.process();
        auto t1 = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING] DAB process() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms" << std::endl;

        auto designRequirements = inputs.get_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

        double magnetizingInductance = 0;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to calculate magnetizing inductance for DAB simulation");
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif

        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        size_t numberOfPeriods = 2;
        if (dabInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = dabInputsJson["numberOfPeriods"].get<size_t>();
        }

        size_t numberOfSteadyStatePeriods = 3;
        if (dabInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = dabInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }

        dabInputs.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        dabInputs.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto t2 = std::chrono::steady_clock::now();
        auto topologyWaveforms = dabInputs.simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        auto t3 = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING] simulate_and_extract_topology_waveforms() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
                  << " ms" << std::endl;

        auto t4 = std::chrono::steady_clock::now();
        auto operatingPoints = dabInputs.simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance);
        auto t5 = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING] simulate_and_extract_operating_points() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count()
                  << " ms" << std::endl;

        json result;

        json inputsJson;
        inputsJson["designRequirements"] = json();
        to_json(inputsJson["designRequirements"], designRequirements);
        inputsJson["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJson["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJson;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // DAB diagnostics (post-simulation, from last solved op point)
        {
            json dabDiag;
            const auto& names = dabInputs.get_per_op_name();
            const auto& mt    = dabInputs.get_per_op_modulation_type();
            const auto& zp    = dabInputs.get_per_op_zvs_margin_primary();
            const auto& zs    = dabInputs.get_per_op_zvs_margin_secondary();
            const auto& d3    = dabInputs.get_per_op_d3_rad();
            const auto& vr    = dabInputs.get_per_op_voltage_conversion_ratio();
            dabDiag["modulationType"]           = mt.empty() ? dabInputs.get_last_modulation_type()           : mt.front();
            dabDiag["computedD3Deg"]            = (d3.empty() ? dabInputs.get_last_d3_rad()                   : d3.front()) * 180.0 / M_PI;
            dabDiag["zvsMarginPrimaryDeg"]      = (zp.empty() ? dabInputs.get_last_zvs_margin_primary()       : zp.front()) * 180.0 / M_PI;
            dabDiag["zvsMarginSecondaryDeg"]    = (zs.empty() ? dabInputs.get_last_zvs_margin_secondary()     : zs.front()) * 180.0 / M_PI;
            dabDiag["computedSeriesInductance"] = dabInputs.get_computed_series_inductance();
            dabDiag["voltageConversionRatio"]   = vr.empty() ? dabInputs.get_last_voltage_conversion_ratio() : vr.front();
            json perOp = json::array();
            for (size_t i = 0; i < mt.size(); ++i) {
                json row;
                row["operatingPointName"]        = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["modulationType"]            = mt[i];
                row["zvsMarginPrimaryDeg"]       = zp[i] * 180.0 / M_PI;
                row["zvsMarginSecondaryDeg"]     = zs[i] * 180.0 / M_PI;
                row["computedD3Deg"]             = d3[i] * 180.0 / M_PI;
                row["voltageConversionRatio"]    = vr[i];
                perOp.push_back(row);
            }
            dabDiag["perOp"] = perOp;
            result["dabDiagnostics"] = dabDiag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ============================================================================
// PSFB / PSHB / AHB - DAB-shaped wizard bindings
// Each pair (calculate_*_inputs / simulate_*_ideal_waveforms) follows the same
// pattern as calculate_dab_inputs / simulate_dab_ideal_waveforms above:
//   - calculate_*: AdvancedX(json).process() -> Inputs -> JSON
//   - simulate_*:  same, then run NgspiceRunner via simulate_and_extract_*
// Returns "Exception: ..." string on failure (calculate path) or
// {"error": "..."} JSON (simulate path) to match DAB / wizard expectations.
// ============================================================================

std::string simulate_psfb_ideal_waveforms(std::string psfbInputsString) {
    try {
        json psfbInputsJson = json::parse(psfbInputsString);

        OpenMagnetics::AdvancedPsfb psfbInputs(psfbInputsJson);
        auto inputs = psfbInputs.process();
        auto designRequirements = inputs.get_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

        double magnetizingInductance = 0;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to determine magnetizing inductance for PSFB simulation");
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        size_t numberOfPeriods = 2;
        if (psfbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = psfbInputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 3;
        if (psfbInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = psfbInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        psfbInputs.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        psfbInputs.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto tA = std::chrono::steady_clock::now();
        auto topologyWaveforms = psfbInputs.simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        auto tB = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING][PSFB] simulate_and_extract_topology_waveforms() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tB - tA).count()
                  << " ms" << std::endl;

        auto tC = std::chrono::steady_clock::now();
        auto operatingPoints = psfbInputs.simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance);
        auto tD = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING][PSFB] simulate_and_extract_operating_points() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tD - tC).count()
                  << " ms" << std::endl;

        json result;
        json inputsJson;
        inputsJson["designRequirements"] = json();
        to_json(inputsJson["designRequirements"], designRequirements);
        inputsJson["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJson["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJson;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // PSFB diagnostics — process() was called above so members are already set.
        {
            json diag;
            const auto& names = psfbInputs.get_per_op_name();
            const auto& dcl   = psfbInputs.get_per_op_duty_cycle_loss();
            const auto& deff  = psfbInputs.get_per_op_effective_duty_cycle();
            const auto& zvs   = psfbInputs.get_per_op_zvs_margin_lagging();
            const auto& zlt   = psfbInputs.get_per_op_zvs_load_threshold();
            const auto& rtt   = psfbInputs.get_per_op_resonant_transition_time();
            const auto& ipk   = psfbInputs.get_per_op_primary_peak_current();
            diag["effectiveDutyCycle"]            = deff.empty() ? psfbInputs.get_last_effective_duty_cycle()     : deff.front();
            diag["dutyCycleLoss"]                 = dcl.empty()  ? psfbInputs.get_last_duty_cycle_loss()          : dcl.front();
            diag["zvsMarginLagging"]              = zvs.empty()  ? psfbInputs.get_last_zvs_margin_lagging()       : zvs.front();
            diag["zvsLoadThreshold"]              = zlt.empty()  ? psfbInputs.get_last_zvs_load_threshold()       : zlt.front();
            diag["resonantTransitionTime"]        = rtt.empty()  ? psfbInputs.get_last_resonant_transition_time() : rtt.front();
            diag["primaryPeakCurrent"]            = ipk.empty()  ? psfbInputs.get_last_primary_peak_current()     : ipk.front();
            diag["computedSeriesInductance"]      = psfbInputs.get_computed_series_inductance();
            diag["computedOutputInductance"]      = psfbInputs.get_computed_output_inductance();
            diag["computedMagnetizingInductance"] = psfbInputs.get_computed_magnetizing_inductance();
            diag["computedDeadTime"]              = psfbInputs.get_computed_dead_time();
            json perOp = json::array();
            for (size_t i = 0; i < dcl.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycleLoss"]         = dcl[i];
                row["effectiveDutyCycle"]    = deff[i];
                row["zvsMarginLagging"]      = zvs[i];
                row["zvsLoadThreshold"]      = zlt[i];
                row["resonantTransitionTime"]= rtt[i];
                row["primaryPeakCurrent"]    = ipk[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["psfbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_pshb_inputs(std::string pshbInputsString) {
    try {
        json pshbInputsJson = json::parse(pshbInputsString);

        OpenMagnetics::AdvancedPshb pshbInputs(pshbInputsJson);

        size_t numberOfPeriods = 1;
        if (pshbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = pshbInputsJson["numberOfPeriods"].get<size_t>();
        }
        pshbInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = pshbInputs.process();

        json result;
        to_json(result, inputs);

        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        // PSHB diagnostics — populated by process() above.
        {
            json diag;
            const auto& names = pshbInputs.get_per_op_name();
            const auto& dcl   = pshbInputs.get_per_op_duty_cycle_loss();
            const auto& deff  = pshbInputs.get_per_op_effective_duty_cycle();
            const auto& zvs   = pshbInputs.get_per_op_zvs_margin_lagging();
            const auto& zlt   = pshbInputs.get_per_op_zvs_load_threshold();
            const auto& rtt   = pshbInputs.get_per_op_resonant_transition_time();
            const auto& ipk   = pshbInputs.get_per_op_primary_peak_current();
            diag["effectiveDutyCycle"]            = deff.empty() ? pshbInputs.get_last_effective_duty_cycle()     : deff.front();
            diag["dutyCycleLoss"]                 = dcl.empty()  ? pshbInputs.get_last_duty_cycle_loss()          : dcl.front();
            diag["zvsMarginLagging"]              = zvs.empty()  ? pshbInputs.get_last_zvs_margin_lagging()       : zvs.front();
            diag["zvsLoadThreshold"]              = zlt.empty()  ? pshbInputs.get_last_zvs_load_threshold()       : zlt.front();
            diag["resonantTransitionTime"]        = rtt.empty()  ? pshbInputs.get_last_resonant_transition_time() : rtt.front();
            diag["primaryPeakCurrent"]            = ipk.empty()  ? pshbInputs.get_last_primary_peak_current()     : ipk.front();
            diag["computedSeriesInductance"]      = pshbInputs.get_computed_series_inductance();
            diag["computedOutputInductance"]      = pshbInputs.get_computed_output_inductance();
            diag["computedMagnetizingInductance"] = pshbInputs.get_computed_magnetizing_inductance();
            diag["computedDeadTime"]              = pshbInputs.get_computed_dead_time();
            json perOp = json::array();
            for (size_t i = 0; i < dcl.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycleLoss"]         = dcl[i];
                row["effectiveDutyCycle"]    = deff[i];
                row["zvsMarginLagging"]      = zvs[i];
                row["zvsLoadThreshold"]      = zlt[i];
                row["resonantTransitionTime"]= rtt[i];
                row["primaryPeakCurrent"]    = ipk[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pshbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_pshb_ideal_waveforms(std::string pshbInputsString) {
    try {
        json pshbInputsJson = json::parse(pshbInputsString);

        OpenMagnetics::AdvancedPshb pshbInputs(pshbInputsJson);
        auto inputs = pshbInputs.process();
        auto designRequirements = inputs.get_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

        double magnetizingInductance = 0;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to determine magnetizing inductance for PSHB simulation");
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        size_t numberOfPeriods = 2;
        if (pshbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = pshbInputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 3;
        if (pshbInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = pshbInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        pshbInputs.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        pshbInputs.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto tA = std::chrono::steady_clock::now();
        auto topologyWaveforms = pshbInputs.simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        auto tB = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING][PSHB] simulate_and_extract_topology_waveforms() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tB - tA).count()
                  << " ms" << std::endl;

        auto tC = std::chrono::steady_clock::now();
        auto operatingPoints = pshbInputs.simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance);
        auto tD = std::chrono::steady_clock::now();
        std::cout << "[WASM-TIMING][PSHB] simulate_and_extract_operating_points() took "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(tD - tC).count()
                  << " ms" << std::endl;

        json result;
        json inputsJson;
        inputsJson["designRequirements"] = json();
        to_json(inputsJson["designRequirements"], designRequirements);
        inputsJson["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJson["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJson;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // PSHB diagnostics — process() was called above so members are already set.
        {
            json diag;
            const auto& names = pshbInputs.get_per_op_name();
            const auto& dcl   = pshbInputs.get_per_op_duty_cycle_loss();
            const auto& deff  = pshbInputs.get_per_op_effective_duty_cycle();
            const auto& zvs   = pshbInputs.get_per_op_zvs_margin_lagging();
            const auto& zlt   = pshbInputs.get_per_op_zvs_load_threshold();
            const auto& rtt   = pshbInputs.get_per_op_resonant_transition_time();
            const auto& ipk   = pshbInputs.get_per_op_primary_peak_current();
            diag["effectiveDutyCycle"]            = deff.empty() ? pshbInputs.get_last_effective_duty_cycle()     : deff.front();
            diag["dutyCycleLoss"]                 = dcl.empty()  ? pshbInputs.get_last_duty_cycle_loss()          : dcl.front();
            diag["zvsMarginLagging"]              = zvs.empty()  ? pshbInputs.get_last_zvs_margin_lagging()       : zvs.front();
            diag["zvsLoadThreshold"]              = zlt.empty()  ? pshbInputs.get_last_zvs_load_threshold()       : zlt.front();
            diag["resonantTransitionTime"]        = rtt.empty()  ? pshbInputs.get_last_resonant_transition_time() : rtt.front();
            diag["primaryPeakCurrent"]            = ipk.empty()  ? pshbInputs.get_last_primary_peak_current()     : ipk.front();
            diag["computedSeriesInductance"]      = pshbInputs.get_computed_series_inductance();
            diag["computedOutputInductance"]      = pshbInputs.get_computed_output_inductance();
            diag["computedMagnetizingInductance"] = pshbInputs.get_computed_magnetizing_inductance();
            diag["computedDeadTime"]              = pshbInputs.get_computed_dead_time();
            json perOp = json::array();
            for (size_t i = 0; i < dcl.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycleLoss"]         = dcl[i];
                row["effectiveDutyCycle"]    = deff[i];
                row["zvsMarginLagging"]      = zvs[i];
                row["zvsLoadThreshold"]      = zlt[i];
                row["resonantTransitionTime"]= rtt[i];
                row["primaryPeakCurrent"]    = ipk[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["pshbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_ahb_inputs(std::string ahbInputsString) {
    try {
        json ahbInputsJson = json::parse(ahbInputsString);

        OpenMagnetics::AdvancedAsymmetricHalfBridge ahbInputs(ahbInputsJson);

        size_t numberOfPeriods = 1;
        if (ahbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = ahbInputsJson["numberOfPeriods"].get<size_t>();
        }
        ahbInputs.set_num_periods_to_extract(numberOfPeriods);

        auto inputs = ahbInputs.process();

        json result;
        to_json(result, inputs);

        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        // AHB diagnostics — populated by process() above.
        {
            json diag;
            const auto& names = ahbInputs.get_per_op_name();
            const auto& dc    = ahbInputs.get_per_op_duty_cycle();
            const auto& cr    = ahbInputs.get_per_op_conversion_ratio();
            const auto& cbv   = ahbInputs.get_per_op_dc_blocking_cap_voltage();
            const auto& cbr   = ahbInputs.get_per_op_dc_blocking_cap_ripple();
            const auto& ppvp  = ahbInputs.get_per_op_primary_peak_voltage_positive();
            const auto& ppvn  = ahbInputs.get_per_op_primary_peak_voltage_negative();
            const auto& spvq1 = ahbInputs.get_per_op_switch_peak_voltage_q1();
            const auto& spvq2 = ahbInputs.get_per_op_switch_peak_voltage_q2();
            const auto& sirq1 = ahbInputs.get_per_op_switch_rms_current_q1();
            const auto& sirq2 = ahbInputs.get_per_op_switch_rms_current_q2();
            const auto& zm    = ahbInputs.get_per_op_zvs_margin();
            const auto& rtt   = ahbInputs.get_per_op_resonant_transition_time();
            const auto& ssfe  = ahbInputs.get_per_op_steady_state_flux_excursion();
            const auto& tfee  = ahbInputs.get_per_op_transient_flux_excursion_estimate();
            const auto& mcr   = ahbInputs.get_per_op_magnetizing_current_ripple();
            const auto& oir   = ahbInputs.get_per_op_output_inductor_ripple();
            const auto& om    = ahbInputs.get_per_op_operating_mode();
            const auto& rt    = ahbInputs.get_per_op_rectifier_type();
            diag["operatingMode"]                = om.empty()    ? ahbInputs.get_last_operating_mode()                     : om.front();
            diag["rectifierType"]                = rt.empty()    ? ahbInputs.get_last_rectifier_type()                     : rt.front();
            diag["dutyCycle"]                    = dc.empty()    ? ahbInputs.get_last_duty_cycle()                         : dc.front();
            diag["conversionRatio"]              = cr.empty()    ? ahbInputs.get_last_conversion_ratio()                   : cr.front();
            diag["dcBlockingCapVoltage"]         = cbv.empty()   ? ahbInputs.get_last_dc_blocking_cap_voltage()             : cbv.front();
            diag["dcBlockingCapRipple"]          = cbr.empty()   ? ahbInputs.get_last_dc_blocking_cap_ripple()              : cbr.front();
            diag["primaryPeakVoltagePositive"]   = ppvp.empty()  ? ahbInputs.get_last_primary_peak_voltage_positive()       : ppvp.front();
            diag["primaryPeakVoltageNegative"]   = ppvn.empty()  ? ahbInputs.get_last_primary_peak_voltage_negative()       : ppvn.front();
            diag["switchPeakVoltageQ1"]          = spvq1.empty() ? ahbInputs.get_last_switch_peak_voltage_q1()              : spvq1.front();
            diag["switchPeakVoltageQ2"]          = spvq2.empty() ? ahbInputs.get_last_switch_peak_voltage_q2()              : spvq2.front();
            diag["switchRmsCurrentQ1"]           = sirq1.empty() ? ahbInputs.get_last_switch_rms_current_q1()               : sirq1.front();
            diag["switchRmsCurrentQ2"]           = sirq2.empty() ? ahbInputs.get_last_switch_rms_current_q2()               : sirq2.front();
            diag["zvsMargin"]                    = zm.empty()    ? ahbInputs.get_last_zvs_margin()                          : zm.front();
            diag["resonantTransitionTime"]       = rtt.empty()   ? ahbInputs.get_last_resonant_transition_time()            : rtt.front();
            diag["steadyStateFluxExcursion"]     = ssfe.empty()  ? ahbInputs.get_last_steady_state_flux_excursion()         : ssfe.front();
            diag["transientFluxExcursionEstimate"] = tfee.empty() ? ahbInputs.get_last_transient_flux_excursion_estimate()  : tfee.front();
            diag["magnetizingCurrentRipple"]     = mcr.empty()   ? ahbInputs.get_last_magnetizing_current_ripple()          : mcr.front();
            diag["outputInductorRipple"]         = oir.empty()   ? ahbInputs.get_last_output_inductor_ripple()              : oir.front();
            json perOp = json::array();
            for (size_t i = 0; i < dc.size(); ++i) {
                json row;
                row["operatingPointName"]            = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]                     = dc[i];
                row["conversionRatio"]               = cr[i];
                row["dcBlockingCapVoltage"]          = cbv[i];
                row["dcBlockingCapRipple"]           = cbr[i];
                row["primaryPeakVoltagePositive"]    = ppvp[i];
                row["primaryPeakVoltageNegative"]    = ppvn[i];
                row["switchPeakVoltageQ1"]           = spvq1[i];
                row["switchPeakVoltageQ2"]           = spvq2[i];
                row["switchRmsCurrentQ1"]            = sirq1[i];
                row["switchRmsCurrentQ2"]            = sirq2[i];
                row["zvsMargin"]                     = zm[i];
                row["resonantTransitionTime"]        = rtt[i];
                row["steadyStateFluxExcursion"]      = ssfe[i];
                row["transientFluxExcursionEstimate"]= tfee[i];
                row["magnetizingCurrentRipple"]      = mcr[i];
                row["outputInductorRipple"]          = oir[i];
                row["operatingMode"]                 = om[i];
                row["rectifierType"]                 = rt[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["ahbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        return "Exception: " + std::string{exc.what()};
    }
}

std::string simulate_ahb_ideal_waveforms(std::string ahbInputsString) {
    try {
        json ahbInputsJson = json::parse(ahbInputsString);

        OpenMagnetics::AdvancedAsymmetricHalfBridge ahbInputs(ahbInputsJson);
        auto inputs = ahbInputs.process();
        auto designRequirements = inputs.get_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

        double magnetizingInductance = 0;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Unable to determine magnetizing inductance for AHB simulation");
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        size_t numberOfPeriods = 2;
        if (ahbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = ahbInputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 3;
        if (ahbInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = ahbInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        ahbInputs.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        ahbInputs.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto topologyWaveforms = ahbInputs.simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        auto operatingPoints = ahbInputs.simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance);

        json result;
        json inputsJson;
        inputsJson["designRequirements"] = json();
        to_json(inputsJson["designRequirements"], designRequirements);
        inputsJson["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJson["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJson;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // AHB diagnostics — process() was called above so members are already set.
        {
            json diag;
            const auto& names = ahbInputs.get_per_op_name();
            const auto& dc    = ahbInputs.get_per_op_duty_cycle();
            const auto& cr    = ahbInputs.get_per_op_conversion_ratio();
            const auto& cbv   = ahbInputs.get_per_op_dc_blocking_cap_voltage();
            const auto& cbr   = ahbInputs.get_per_op_dc_blocking_cap_ripple();
            const auto& ppvp  = ahbInputs.get_per_op_primary_peak_voltage_positive();
            const auto& ppvn  = ahbInputs.get_per_op_primary_peak_voltage_negative();
            const auto& spvq1 = ahbInputs.get_per_op_switch_peak_voltage_q1();
            const auto& spvq2 = ahbInputs.get_per_op_switch_peak_voltage_q2();
            const auto& sirq1 = ahbInputs.get_per_op_switch_rms_current_q1();
            const auto& sirq2 = ahbInputs.get_per_op_switch_rms_current_q2();
            const auto& zm    = ahbInputs.get_per_op_zvs_margin();
            const auto& rtt   = ahbInputs.get_per_op_resonant_transition_time();
            const auto& ssfe  = ahbInputs.get_per_op_steady_state_flux_excursion();
            const auto& tfee  = ahbInputs.get_per_op_transient_flux_excursion_estimate();
            const auto& mcr   = ahbInputs.get_per_op_magnetizing_current_ripple();
            const auto& oir   = ahbInputs.get_per_op_output_inductor_ripple();
            const auto& om    = ahbInputs.get_per_op_operating_mode();
            const auto& rt    = ahbInputs.get_per_op_rectifier_type();
            diag["operatingMode"]                = om.empty()    ? ahbInputs.get_last_operating_mode()                     : om.front();
            diag["rectifierType"]                = rt.empty()    ? ahbInputs.get_last_rectifier_type()                     : rt.front();
            diag["dutyCycle"]                    = dc.empty()    ? ahbInputs.get_last_duty_cycle()                         : dc.front();
            diag["conversionRatio"]              = cr.empty()    ? ahbInputs.get_last_conversion_ratio()                   : cr.front();
            diag["dcBlockingCapVoltage"]         = cbv.empty()   ? ahbInputs.get_last_dc_blocking_cap_voltage()             : cbv.front();
            diag["dcBlockingCapRipple"]          = cbr.empty()   ? ahbInputs.get_last_dc_blocking_cap_ripple()              : cbr.front();
            diag["primaryPeakVoltagePositive"]   = ppvp.empty()  ? ahbInputs.get_last_primary_peak_voltage_positive()       : ppvp.front();
            diag["primaryPeakVoltageNegative"]   = ppvn.empty()  ? ahbInputs.get_last_primary_peak_voltage_negative()       : ppvn.front();
            diag["switchPeakVoltageQ1"]          = spvq1.empty() ? ahbInputs.get_last_switch_peak_voltage_q1()              : spvq1.front();
            diag["switchPeakVoltageQ2"]          = spvq2.empty() ? ahbInputs.get_last_switch_peak_voltage_q2()              : spvq2.front();
            diag["switchRmsCurrentQ1"]           = sirq1.empty() ? ahbInputs.get_last_switch_rms_current_q1()               : sirq1.front();
            diag["switchRmsCurrentQ2"]           = sirq2.empty() ? ahbInputs.get_last_switch_rms_current_q2()               : sirq2.front();
            diag["zvsMargin"]                    = zm.empty()    ? ahbInputs.get_last_zvs_margin()                          : zm.front();
            diag["resonantTransitionTime"]       = rtt.empty()   ? ahbInputs.get_last_resonant_transition_time()            : rtt.front();
            diag["steadyStateFluxExcursion"]     = ssfe.empty()  ? ahbInputs.get_last_steady_state_flux_excursion()         : ssfe.front();
            diag["transientFluxExcursionEstimate"] = tfee.empty() ? ahbInputs.get_last_transient_flux_excursion_estimate()  : tfee.front();
            diag["magnetizingCurrentRipple"]     = mcr.empty()   ? ahbInputs.get_last_magnetizing_current_ripple()          : mcr.front();
            diag["outputInductorRipple"]         = oir.empty()   ? ahbInputs.get_last_output_inductor_ripple()              : oir.front();
            json perOp = json::array();
            for (size_t i = 0; i < dc.size(); ++i) {
                json row;
                row["operatingPointName"]            = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"]                     = dc[i];
                row["conversionRatio"]               = cr[i];
                row["dcBlockingCapVoltage"]          = cbv[i];
                row["dcBlockingCapRipple"]           = cbr[i];
                row["primaryPeakVoltagePositive"]    = ppvp[i];
                row["primaryPeakVoltageNegative"]    = ppvn[i];
                row["switchPeakVoltageQ1"]           = spvq1[i];
                row["switchPeakVoltageQ2"]           = spvq2[i];
                row["switchRmsCurrentQ1"]            = sirq1[i];
                row["switchRmsCurrentQ2"]            = sirq2[i];
                row["zvsMargin"]                     = zm[i];
                row["resonantTransitionTime"]        = rtt[i];
                row["steadyStateFluxExcursion"]      = ssfe[i];
                row["transientFluxExcursionEstimate"]= tfee[i];
                row["magnetizingCurrentRipple"]      = mcr[i];
                row["outputInductorRipple"]          = oir[i];
                row["operatingMode"]                 = om[i];
                row["rectifierType"]                 = rt[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["ahbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_cllc_inputs(std::string cllcInputsString) {
    try {
        json cllcInputsJson = json::parse(cllcInputsString);
        
        // Check for multi-output request (not yet supported for CLLC)
        if (cllcInputsJson.contains("operatingPoints") && cllcInputsJson["operatingPoints"].is_array()) {
            for (const auto& op : cllcInputsJson["operatingPoints"]) {
                if (op.contains("outputVoltages") && op["outputVoltages"].is_array() && op["outputVoltages"].size() > 1) {
                    throw std::runtime_error("Multi-output configuration is not yet supported for CLLC converter. Please use a single output (outputVoltages array with one element).");
                }
                if (op.contains("outputCurrents") && op["outputCurrents"].is_array() && op["outputCurrents"].size() > 1) {
                    throw std::runtime_error("Multi-output configuration is not yet supported for CLLC converter. Please use a single output (outputCurrents array with one element).");
                }
            }
        }
        
        OpenMagnetics::CllcConverter cllcInputs(cllcInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (cllcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = cllcInputsJson["numberOfPeriods"].get<size_t>();
        }
        cllcInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto designRequirements = cllcInputs.process_design_requirements();

        // Extract turns ratios from designRequirements (set by process_design_requirements()).
        // process_operating_points(turnsRatios, Lm) requires a non-empty turnsRatios
        // vector — it dereferences turnsRatios[0] unconditionally. If desiredTurnsRatios
        // was supplied, the JSON ctor will have set it on designRequirements; otherwise
        // we use the value calculate_resonant_parameters() computed.
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("CLLC: process_design_requirements produced no turns ratios");
        }

        // Magnetizing inductance: prefer explicit user input, else use the
        // value computed by calculate_resonant_parameters() (stored on the
        // designRequirements). Mirrors LLC.process() behaviour.
        double magnetizingInductance;
        if (cllcInputsJson.contains("magnetizingInductance") &&
            cllcInputsJson["magnetizingInductance"].is_number()) {
            magnetizingInductance = cllcInputsJson["magnetizingInductance"].get<double>();
        }
        else {
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("CLLC: no magnetizing inductance available (neither user input nor computed)");
            }
        }

        auto operatingPoints = cllcInputs.process_operating_points(turnsRatios, magnetizingInductance);
        
        // Commented out due to missing calculate_resonant_tank method
        // if (!cllcInputsJson.contains("primaryResonantInductance")) {
        //     double ls1, cs1, ls2, cs2, lm;
        //     double fr = (cllcInputs.get_min_switching_frequency() + cllcInputs.get_max_switching_frequency()) / 2.0;
        //     cllcInputs.calculate_resonant_tank(fr, cllcInputs.get_quality_factor().value_or(0.4), ls1, cs1, ls2, cs2, lm);
        // }
        
        json result;
        // Match LLC's calculate_llc_inputs output shape: { designRequirements, operatingPoints }
        // The frontend ConverterWizardBase.processWizardData reads r.designRequirements;
        // a flattened DR at the root would make that field undefined and break
        // Review Specs / Design Magnetic navigation (setupMasStore would throw
        // "Cannot set properties of undefined (setting 'topology')").
        json drJson;
        to_json(drJson, designRequirements);
        result["designRequirements"] = drJson;
        result["operatingPoints"] = json::array();

        for (auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            result["operatingPoints"].push_back(opJson);
        }

        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        // CLLC diagnostics — populated by process_operating_point_for_input_voltage
        // (called inside process_operating_points above). Mirrors simulate_cllc_ideal_waveforms.
        json diag;
        diag["lipFrequency"]               = cllcInputs.get_lip_frequency();
        diag["lastSubStateSequence"]       = cllcInputs.get_last_sub_state_sequence();
        diag["bridgeVoltageFactor"]        = cllcInputs.get_bridge_voltage_factor();
        {
            const auto& names = cllcInputs.get_per_op_name();
            const auto& mode  = cllcInputs.get_per_op_mode();
            const auto& res   = cllcInputs.get_per_op_steady_state_residual();
            const auto& zp    = cllcInputs.get_per_op_zvs_margin_primary();
            const auto& zs    = cllcInputs.get_per_op_zvs_margin_secondary();
            const auto& rtt   = cllcInputs.get_per_op_resonant_transition_time();
            const auto& ipk   = cllcInputs.get_per_op_primary_peak_current();
            const auto& vcr   = cllcInputs.get_per_op_resonant_cap_peak_voltage();
            diag["lastMode"]                   = mode.empty() ? cllcInputs.get_last_mode()                       : mode.front();
            diag["lastSteadyStateResidual"]    = res.empty()  ? cllcInputs.get_last_steady_state_residual()      : res.front();
            diag["lastZvsMarginPrimary"]       = zp.empty()   ? cllcInputs.get_last_zvs_margin_primary()         : zp.front();
            diag["lastZvsMarginSecondary"]     = zs.empty()   ? cllcInputs.get_last_zvs_margin_secondary()       : zs.front();
            diag["lastResonantTransitionTime"] = rtt.empty()  ? cllcInputs.get_last_resonant_transition_time()   : rtt.front();
            diag["lastPrimaryPeakCurrent"]     = ipk.empty()  ? cllcInputs.get_last_primary_peak_current()       : ipk.front();
            diag["lastResonantCapPeakVoltage"] = vcr.empty()  ? cllcInputs.get_last_resonant_cap_peak_voltage()  : vcr.front();
            json perOp = json::array();
            for (size_t i = 0; i < mode.size(); ++i) {
                json row;
                row["operatingPointName"]      = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["lastMode"]                = mode[i];
                row["steadyStateResidual"]     = res[i];
                row["zvsMarginPrimary"]        = zp[i];
                row["zvsMarginSecondary"]      = zs[i];
                row["resonantTransitionTime"]  = rtt[i];
                row["primaryPeakCurrent"]      = ipk[i];
                row["resonantCapPeakVoltage"]  = vcr[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
        }
        result["cllcDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// SPICE/TDA-based CLLC ideal waveform simulator. Mirrors
// simulate_llc_ideal_waveforms (7817): runs both topology-waveform and
// per-OP extractions through ngspice, returns inputs + converterWaveforms
// + cllcDiagnostics (richer than LLC — includes ZVS margins, resonant
// transition time, peak primary current, peak resonant-cap voltage, and
// the per-segment sub-state sequence).
std::string simulate_cllc_ideal_waveforms(std::string cllcInputsString) {
    try {
        json cllcInputsJson = json::parse(cllcInputsString);

        // Reject multi-output (CLLC backend is single-output only).
        if (cllcInputsJson.contains("operatingPoints") && cllcInputsJson["operatingPoints"].is_array()) {
            for (const auto& op : cllcInputsJson["operatingPoints"]) {
                if (op.contains("outputVoltages") && op["outputVoltages"].is_array() && op["outputVoltages"].size() > 1) {
                    throw std::runtime_error("Multi-output configuration is not yet supported for CLLC converter.");
                }
                if (op.contains("outputCurrents") && op["outputCurrents"].is_array() && op["outputCurrents"].size() > 1) {
                    throw std::runtime_error("Multi-output configuration is not yet supported for CLLC converter.");
                }
            }
        }

        OpenMagnetics::CllcConverter cllc(cllcInputsJson);

        auto designRequirements = cllc.process_design_requirements();

        // Turns ratios from design requirements (single secondary expected).
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("CLLC: process_design_requirements produced no turns ratios");
        }

        // Magnetizing inductance: prefer explicit user input, else use the
        // value computed by process_design_requirements().
        double magnetizingInductance;
        if (cllcInputsJson.contains("magnetizingInductance") &&
            cllcInputsJson["magnetizingInductance"].is_number()) {
            magnetizingInductance = cllcInputsJson["magnetizingInductance"].get<double>();
        }
        else {
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("CLLC: no magnetizing inductance available (neither user input nor computed)");
            }
        }

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif

        OpenMagnetics::NgspiceRunner runner;
        if (!runner.is_available()) {
            throw std::runtime_error("ngspice simulation is required but ngspice is not available");
        }

        size_t numberOfPeriods = 2;
        if (cllcInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = cllcInputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 3;
        if (cllcInputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = cllcInputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        cllc.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        cllc.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto topologyWaveforms = cllc.simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        // Note: CllcConverter::simulate_and_extract_operating_points has no
        // numberOfPeriods argument (CLLC API choice — operating-point extraction
        // always uses the converter's stored num_periods_to_extract setter,
        // which we just set above).
        auto operatingPoints = cllc.simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance);

        // The SPICE path never calls process_operating_point_for_input_voltage,
        // so diagnostic members (LIP freq, ZVS margins, peak currents, etc.) stay
        // at their default 0. Run the analytical solver now to populate them —
        // we discard its operating-point return value and keep the SPICE waveforms.
        cllc.process_operating_points(turnsRatios, magnetizingInductance);

        json result;

        json inputsJson;
        inputsJson["designRequirements"] = json();
        to_json(inputsJson["designRequirements"], designRequirements);
        inputsJson["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJson["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJson;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        // CLLC diagnostics — richer than LLC's. Mirrors plan §5.4.
        json diag;
        diag["lipFrequency"]              = cllc.get_lip_frequency();
        diag["lastSubStateSequence"]      = cllc.get_last_sub_state_sequence();
        diag["bridgeVoltageFactor"]       = cllc.get_bridge_voltage_factor();
        {
            const auto& names = cllc.get_per_op_name();
            const auto& mode  = cllc.get_per_op_mode();
            const auto& res   = cllc.get_per_op_steady_state_residual();
            const auto& zp    = cllc.get_per_op_zvs_margin_primary();
            const auto& zs    = cllc.get_per_op_zvs_margin_secondary();
            const auto& rtt   = cllc.get_per_op_resonant_transition_time();
            const auto& ipk   = cllc.get_per_op_primary_peak_current();
            const auto& vcr   = cllc.get_per_op_resonant_cap_peak_voltage();
            diag["lastMode"]                  = mode.empty() ? cllc.get_last_mode()                      : mode.front();
            diag["lastSteadyStateResidual"]   = res.empty()  ? cllc.get_last_steady_state_residual()     : res.front();
            diag["lastZvsMarginPrimary"]      = zp.empty()   ? cllc.get_last_zvs_margin_primary()        : zp.front();
            diag["lastZvsMarginSecondary"]    = zs.empty()   ? cllc.get_last_zvs_margin_secondary()      : zs.front();
            diag["lastResonantTransitionTime"]= rtt.empty()  ? cllc.get_last_resonant_transition_time()  : rtt.front();
            diag["lastPrimaryPeakCurrent"]    = ipk.empty()  ? cllc.get_last_primary_peak_current()      : ipk.front();
            diag["lastResonantCapPeakVoltage"]= vcr.empty()  ? cllc.get_last_resonant_cap_peak_voltage() : vcr.front();
            json perOp = json::array();
            for (size_t i = 0; i < mode.size(); ++i) {
                json row;
                row["operatingPointName"]     = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["lastMode"]               = mode[i];
                row["steadyStateResidual"]    = res[i];
                row["zvsMarginPrimary"]       = zp[i];
                row["zvsMarginSecondary"]     = zs[i];
                row["resonantTransitionTime"] = rtt[i];
                row["primaryPeakCurrent"]     = ipk[i];
                row["resonantCapPeakVoltage"] = vcr[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
        }
        result["cllcDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_psfb_inputs(std::string psfbInputsString) {
    try {
        json psfbInputsJson = json::parse(psfbInputsString);
        
        // Check for multi-output request (not yet supported for PSFB)
        if (psfbInputsJson.contains("operatingPoints") && psfbInputsJson["operatingPoints"].is_array()) {
            for (const auto& op : psfbInputsJson["operatingPoints"]) {
                if (op.contains("outputVoltages") && op["outputVoltages"].is_array() && op["outputVoltages"].size() > 1) {
                    throw std::runtime_error("Multi-output configuration is not yet supported for PSFB converter. Please use a single output (outputVoltages array with one element).");
                }
                if (op.contains("outputCurrents") && op["outputCurrents"].is_array() && op["outputCurrents"].size() > 1) {
                    throw std::runtime_error("Multi-output configuration is not yet supported for PSFB converter. Please use a single output (outputCurrents array with one element).");
                }
            }
        }
        
        OpenMagnetics::Psfb psfbInputs(psfbInputsJson);
        
        // Read number of periods from input (default to 1 for analytical)
        size_t numberOfPeriods = 1;
        if (psfbInputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = psfbInputsJson["numberOfPeriods"].get<size_t>();
        }
        psfbInputs.set_num_periods_to_extract(numberOfPeriods);
        
        auto designRequirements = psfbInputs.process_design_requirements();

        // Extract turns ratios from designRequirements (set by process_design_requirements()).
        // process_operating_points(turnsRatios, Lm) dereferences turnsRatios[0] —
        // passing an empty vector produces NaN waveforms downstream.
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("PSFB: process_design_requirements produced no turns ratios");
        }

        // Magnetizing inductance: prefer explicit user input, else use computed value.
        double magnetizingInductance;
        if (psfbInputsJson.contains("magnetizingInductance") &&
            psfbInputsJson["magnetizingInductance"].is_number()) {
            magnetizingInductance = psfbInputsJson["magnetizingInductance"].get<double>();
        }
        else {
            magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
            if (!(magnetizingInductance > 0)) {
                throw std::runtime_error("PSFB: no magnetizing inductance available (neither user input nor computed)");
            }
        }

        auto operatingPoints = psfbInputs.process_operating_points(turnsRatios, magnetizingInductance);
        
        json result;
        // Nest designRequirements (matches LLC/CLLC; ConverterWizardBase
        // reads result.designRequirements — a flattened DR at the root would
        // make that undefined and break Review Specs / Design Magnetic
        // navigation with "Cannot set properties of undefined (setting 'topology')").
        json drJson;
        to_json(drJson, designRequirements);
        result["designRequirements"] = drJson;
        result["operatingPoints"] = json::array();
        
        for (auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            result["operatingPoints"].push_back(opJson);
        }
        
        // Repeat waveforms for the specified number of periods (analytical generates 1 period)
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        // PSFB diagnostics — populated by process_operating_points above.
        {
            json diag;
            const auto& names = psfbInputs.get_per_op_name();
            const auto& dcl   = psfbInputs.get_per_op_duty_cycle_loss();
            const auto& deff  = psfbInputs.get_per_op_effective_duty_cycle();
            const auto& zvs   = psfbInputs.get_per_op_zvs_margin_lagging();
            const auto& zlt   = psfbInputs.get_per_op_zvs_load_threshold();
            const auto& rtt   = psfbInputs.get_per_op_resonant_transition_time();
            const auto& ipk   = psfbInputs.get_per_op_primary_peak_current();
            diag["effectiveDutyCycle"]          = deff.empty() ? psfbInputs.get_last_effective_duty_cycle()     : deff.front();
            diag["dutyCycleLoss"]               = dcl.empty()  ? psfbInputs.get_last_duty_cycle_loss()          : dcl.front();
            diag["zvsMarginLagging"]            = zvs.empty()  ? psfbInputs.get_last_zvs_margin_lagging()       : zvs.front();
            diag["zvsLoadThreshold"]            = zlt.empty()  ? psfbInputs.get_last_zvs_load_threshold()       : zlt.front();
            diag["resonantTransitionTime"]      = rtt.empty()  ? psfbInputs.get_last_resonant_transition_time() : rtt.front();
            diag["primaryPeakCurrent"]          = ipk.empty()  ? psfbInputs.get_last_primary_peak_current()     : ipk.front();
            diag["computedSeriesInductance"]    = psfbInputs.get_computed_series_inductance();
            diag["computedOutputInductance"]    = psfbInputs.get_computed_output_inductance();
            diag["computedMagnetizingInductance"] = psfbInputs.get_computed_magnetizing_inductance();
            diag["computedDeadTime"]            = psfbInputs.get_computed_dead_time();
            json perOp = json::array();
            for (size_t i = 0; i < dcl.size(); ++i) {
                json row;
                row["operatingPointName"]    = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycleLoss"]         = dcl[i];
                row["effectiveDutyCycle"]    = deff[i];
                row["zvsMarginLagging"]      = zvs[i];
                row["zvsLoadThreshold"]      = zlt[i];
                row["resonantTransitionTime"]= rtt[i];
                row["primaryPeakCurrent"]    = ipk[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["psfbDiagnostics"] = diag;
        }

        result["masInputs"] = json::object();
        to_json(result["masInputs"], designRequirements);
        if (!result["operatingPoints"].empty()) {
            result["masInputs"]["operatingPoints"] = json::array({result["operatingPoints"][0]});
        } else {
            result["masInputs"]["operatingPoints"] = json::array();
        }
        
        return result.dump(4);
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// New integrated functions for topology processing and magnetic advising

std::string process_converter(std::string topologyName, std::string converterJson, bool useNgspice) {
    try {
        json converterData = json::parse(converterJson);
        json result;
        
        // Normalize topology name to lowercase
        std::string topology = topologyName;
        std::transform(topology.begin(), topology.end(), topology.begin(), ::tolower);
        
        if (topology == "flyback" || topology == "advanced_flyback") {
            bool isAdvanced = converterData.contains("desiredInductance");
            if (useNgspice) {
                // simulate_flyback_ideal_waveforms handles both regular and advanced internally
                return simulate_flyback_ideal_waveforms(converterJson);
            } else {
                if (isAdvanced) {
                    return calculate_advanced_flyback_inputs(converterJson);
                } else {
                    return calculate_flyback_inputs(converterJson);
                }
            }
        }
        else if (topology == "buck" || topology == "advanced_buck") {
            if (useNgspice) {
                return simulate_buck_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_buck_inputs(converterJson);
                } else {
                    return calculate_buck_inputs(converterJson);
                }
            }
        }
        else if (topology == "boost" || topology == "advanced_boost") {
            if (useNgspice) {
                return simulate_boost_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_boost_inputs(converterJson);
                } else {
                    return calculate_boost_inputs(converterJson);
                }
            }
        }
        else if (topology == "sepic" || topology == "advanced_sepic") {
            if (useNgspice) {
                return simulate_sepic_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_sepic_inputs(converterJson);
                } else {
                    return calculate_sepic_inputs(converterJson);
                }
            }
        }
        else if (topology == "isolated_buck" || topology == "advanced_isolated_buck") {
            if (useNgspice) {
                return simulate_isolated_buck_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_isolated_buck_inputs(converterJson);
                } else {
                    return calculate_isolated_buck_inputs(converterJson);
                }
            }
        }
        else if (topology == "isolated_buck_boost" || topology == "advanced_isolated_buck_boost") {
            if (useNgspice) {
                return simulate_isolated_buck_boost_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_isolated_buck_boost_inputs(converterJson);
                } else {
                    return calculate_isolated_buck_boost_inputs(converterJson);
                }
            }
        }
        else if (topology == "push_pull" || topology == "advanced_push_pull") {
            if (useNgspice) {
                return simulate_push_pull_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_push_pull_inputs(converterJson);
                } else {
                    return calculate_push_pull_inputs(converterJson);
                }
            }
        }
        else if (topology == "single_switch_forward" || topology == "advanced_single_switch_forward") {
            if (useNgspice) {
                return simulate_forward_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_single_switch_forward_inputs(converterJson);
                } else {
                    return calculate_single_switch_forward_inputs(converterJson);
                }
            }
        }
        else if (topology == "two_switch_forward" || topology == "advanced_two_switch_forward") {
            if (useNgspice) {
                return simulate_two_switch_forward_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_two_switch_forward_inputs(converterJson);
                } else {
                    return calculate_two_switch_forward_inputs(converterJson);
                }
            }
        }
        else if (topology == "active_clamp_forward" || topology == "advanced_active_clamp_forward") {
            if (useNgspice) {
                return simulate_active_clamp_forward_ideal_waveforms(converterJson);
            } else {
                if (converterData.contains("desiredInductance")) {
                    return calculate_advanced_active_clamp_forward_inputs(converterJson);
                } else {
                    return calculate_active_clamp_forward_inputs(converterJson);
                }
            }
        }
        else if (topology == "llc" || topology == "advanced_llc") {
            if (useNgspice) {
                return simulate_llc_ideal_waveforms(converterJson);
            } else {
                return calculate_llc_inputs(converterJson);
            }
        }
        else if (topology == "cllc" || topology == "advanced_cllc") {
            if (useNgspice) {
                return simulate_cllc_ideal_waveforms(converterJson);
            } else {
                return calculate_cllc_inputs(converterJson);
            }
        }
        else if (topology == "dab" || topology == "advanced_dab") {
            return calculate_dab_inputs(converterJson);
        }
        else if (topology == "psfb" || topology == "phase_shifted_full_bridge" || 
                 topology == "advanced_psfb" || topology == "advanced_phase_shifted_full_bridge") {
            return calculate_psfb_inputs(converterJson);
        }
        else if (topology == "clllc" || topology == "advanced_clllc") {
            if (useNgspice) {
                return simulate_clllc_ideal_waveforms(converterJson);
            } else {
                return calculate_clllc_inputs(converterJson);
            }
        }
        else if (topology == "cuk" || topology == "advanced_cuk") {
            if (useNgspice) {
                return simulate_cuk_ideal_waveforms(converterJson);
            } else {
                return calculate_cuk_inputs(converterJson);
            }
        }
        else if (topology == "four_switch_buck_boost" || topology == "advanced_four_switch_buck_boost") {
            if (useNgspice) {
                return simulate_four_switch_buck_boost_ideal_waveforms(converterJson);
            } else {
                return calculate_four_switch_buck_boost_inputs(converterJson);
            }
        }
        else if (topology == "weinberg" || topology == "advanced_weinberg") {
            if (useNgspice) {
                return simulate_weinberg_ideal_waveforms(converterJson);
            } else {
                return calculate_weinberg_inputs(converterJson);
            }
        }
        else if (topology == "zeta" || topology == "advanced_zeta") {
            if (useNgspice) {
                return simulate_zeta_ideal_waveforms(converterJson);
            } else {
                return calculate_zeta_inputs(converterJson);
            }
        }
        else {
            json error;
            error["error"] = "Unknown topology: " + topologyName;
            return error.dump(4);
        }
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string design_magnetics_from_converter(std::string topologyName, std::string converterJson, 
                                             int maxResults, std::string coreModeString, 
                                             bool useNgspice, std::string weightsString) {
    try {
        // Step 1: Process the converter to get Inputs
        std::string inputsResult = process_converter(topologyName, converterJson, useNgspice);
        json inputsJson = json::parse(inputsResult);
        
        if (inputsJson.contains("error")) {
            return inputsResult;  // Return the error
        }
        
        // Step 2: Process inputs to ensure they're properly formatted
        OpenMagnetics::Inputs inputs(inputsJson, true);
        
        // Step 3: Set up magnetic adviser
        OpenMagnetics::MagneticAdviser magneticAdviser;
        
        // Parse core mode
        OpenMagnetics::CoreAdviser::CoreAdviserModes coreMode;
        OpenMagnetics::from_json(coreModeString, coreMode);
        magneticAdviser.set_core_mode(coreMode);
        
        // Parse weights if provided
        std::map<OpenMagnetics::MagneticFilters, double> weights;
        if (!weightsString.empty() && weightsString != "{}") {
            std::map<std::string, double> weightsKeysString = json::parse(weightsString);
            double externalSum = 0;
            for (const auto& pair : weightsKeysString) {
                externalSum += pair.second;
            }
            for (const auto& [filterName, weight] : weightsKeysString) {
                OpenMagnetics::MagneticFilters filter;
                OpenMagnetics::from_json(filterName, filter);
                weights[filter] = weight / externalSum;
            }
        }
        
        // Step 4: Get advised magnetics
        std::vector<std::pair<OpenMagnetics::Mas, double>> masMagnetics;
        if (weights.empty()) {
            masMagnetics = magneticAdviser.get_advised_magnetic(inputs, maxResults);
        } else {
            masMagnetics = magneticAdviser.get_advised_magnetic(inputs, weights, maxResults);
        }
        
        // Step 5: Build result
        auto scorings = magneticAdviser.get_scorings();
        json results;
        results["data"] = json::array();
        
        for (auto& [masMagnetic, scoring] : masMagnetics) {
            std::string name = masMagnetic.get_magnetic().get_manufacturer_info().value().get_reference().value();
            
            json result;
            json masJson;
            to_json(masJson, masMagnetic);
            result["mas"] = masJson;
            result["scoring"] = scoring;
            
            // Add scoring per filter if available
            if (scorings.count(name)) {
                result["scoringPerFilter"] = json();
                for (size_t index = 0; index < magic_enum::enum_count<OpenMagnetics::MagneticFilters>(); ++index) {
                    auto filter = static_cast<OpenMagnetics::MagneticFilters>(index);
                    auto filterString = OpenMagnetics::to_string(filter);
                    if (scorings[name].count(filter)) {
                        result["scoringPerFilter"][filterString] = scorings[name][filter];
                    }
                }
            }
            
            results["data"].push_back(result);
        }
        
        // Sort by scoring
        std::sort(results["data"].begin(), results["data"].end(), 
                  [](json& b1, json& b2) {
                      return b1["scoring"] > b2["scoring"];
                  });
        
        return results.dump(4);
    }
    catch (const std::exception &exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

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
std::string calculate_clllc_inputs(std::string clllcInputsString) {
    try {
        json inputsJson = json::parse(clllcInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::Clllc model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            diag["computedPrimarySeriesInductance"]      = model.get_computed_primary_series_inductance();
            diag["computedSecondarySeriesInductance"]    = model.get_computed_secondary_series_inductance();
            diag["computedPrimaryResonantCapacitance"]   = model.get_computed_primary_resonant_capacitance();
            diag["computedSecondaryResonantCapacitance"] = model.get_computed_secondary_resonant_capacitance();
            diag["computedMagnetizingInductance"]        = model.get_computed_magnetizing_inductance();
            diag["computedTurnsRatio"]                   = model.get_computed_turns_ratio();
            diag["computedDeadTime"]                     = model.get_computed_dead_time();
            diag["computedInductanceRatioK"]             = model.get_computed_inductance_ratio_k();
            diag["computedQualityFactor"]                = model.get_computed_quality_factor();
            diag["computedPrimaryResonantFrequency"]     = model.get_computed_primary_resonant_frequency();
            {
                const auto& names = model.get_per_op_name();
                const auto& mf    = model.get_per_op_mode_forward();
                const auto& mr    = model.get_per_op_mode_reverse();
                const auto& zpl   = model.get_per_op_zvs_margin_primary_lagging();
                const auto& zsl   = model.get_per_op_zvs_margin_secondary_lagging();
                const auto& zltp  = model.get_per_op_zvs_load_threshold_primary();
                const auto& zlts  = model.get_per_op_zvs_load_threshold_secondary();
                const auto& rtt   = model.get_per_op_resonant_transition_time();
                const auto& ipk   = model.get_per_op_primary_peak_current();
                const auto& spk   = model.get_per_op_secondary_peak_current();
                const auto& irms  = model.get_per_op_primary_rms_current();
                const auto& srms  = model.get_per_op_secondary_rms_current();
                const auto& mpk   = model.get_per_op_magnetizing_peak_current();
                const auto& vc1   = model.get_per_op_cr1_peak_voltage();
                const auto& vc2   = model.get_per_op_cr2_peak_voltage();
                const auto& shr   = model.get_per_op_current_sharing_ratio();
                const auto& res   = model.get_per_op_steady_state_residual();
                diag["lastModeForward"]                = mf.empty()   ? model.get_last_mode_forward()                  : mf.front();
                diag["lastModeReverse"]                = mr.empty()   ? model.get_last_mode_reverse()                  : mr.front();
                diag["lastZvsMarginPrimaryLagging"]    = zpl.empty()  ? model.get_last_zvs_margin_primary_lagging()    : zpl.front();
                diag["lastZvsMarginSecondaryLagging"]  = zsl.empty()  ? model.get_last_zvs_margin_secondary_lagging()  : zsl.front();
                diag["lastZvsLoadThresholdPrimary"]    = zltp.empty() ? model.get_last_zvs_load_threshold_primary()    : zltp.front();
                diag["lastZvsLoadThresholdSecondary"]  = zlts.empty() ? model.get_last_zvs_load_threshold_secondary()  : zlts.front();
                diag["lastResonantTransitionTime"]     = rtt.empty()  ? model.get_last_resonant_transition_time()      : rtt.front();
                diag["lastPrimaryPeakCurrent"]         = ipk.empty()  ? model.get_last_primary_peak_current()          : ipk.front();
                diag["lastSecondaryPeakCurrent"]       = spk.empty()  ? model.get_last_secondary_peak_current()        : spk.front();
                diag["lastPrimaryRmsCurrent"]          = irms.empty() ? model.get_last_primary_rms_current()           : irms.front();
                diag["lastSecondaryRmsCurrent"]        = srms.empty() ? model.get_last_secondary_rms_current()         : srms.front();
                diag["lastMagnetizingPeakCurrent"]     = mpk.empty()  ? model.get_last_magnetizing_peak_current()      : mpk.front();
                diag["lastCr1PeakVoltage"]             = vc1.empty()  ? model.get_last_cr1_peak_voltage()              : vc1.front();
                diag["lastCr2PeakVoltage"]             = vc2.empty()  ? model.get_last_cr2_peak_voltage()              : vc2.front();
                diag["lastCurrentSharingRatio"]        = shr.empty()  ? model.get_last_current_sharing_ratio()         : shr.front();
                diag["lastSteadyStateResidual"]        = res.empty()  ? model.get_last_steady_state_residual()         : res.front();
                json perOp = json::array();
                for (size_t i = 0; i < mf.size(); ++i) {
                    json row;
                    row["operatingPointName"]              = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                    row["modeForward"]                     = mf[i];
                    row["modeReverse"]                     = mr[i];
                    row["zvsMarginPrimaryLagging"]         = zpl[i];
                    row["zvsMarginSecondaryLagging"]       = zsl[i];
                    row["zvsLoadThresholdPrimary"]         = zltp[i];
                    row["zvsLoadThresholdSecondary"]       = zlts[i];
                    row["resonantTransitionTime"]          = rtt[i];
                    row["primaryPeakCurrent"]              = ipk[i];
                    row["secondaryPeakCurrent"]            = spk[i];
                    row["primaryRmsCurrent"]               = irms[i];
                    row["secondaryRmsCurrent"]             = srms[i];
                    row["magnetizingPeakCurrent"]          = mpk[i];
                    row["cr1PeakVoltage"]                  = vc1[i];
                    row["cr2PeakVoltage"]                  = vc2[i];
                    row["currentSharingRatio"]             = shr[i];
                    row["steadyStateResidual"]             = res[i];
                    perOp.push_back(row);
                }
                diag["perOp"] = perOp;
            }
            result["clllcDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_advanced_clllc_inputs(std::string clllcInputsString) {
    try {
        json inputsJson = json::parse(clllcInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::AdvancedClllc model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            diag["computedPrimarySeriesInductance"]      = model.get_computed_primary_series_inductance();
            diag["computedSecondarySeriesInductance"]    = model.get_computed_secondary_series_inductance();
            diag["computedPrimaryResonantCapacitance"]   = model.get_computed_primary_resonant_capacitance();
            diag["computedSecondaryResonantCapacitance"] = model.get_computed_secondary_resonant_capacitance();
            diag["computedMagnetizingInductance"]        = model.get_computed_magnetizing_inductance();
            diag["computedTurnsRatio"]                   = model.get_computed_turns_ratio();
            diag["computedDeadTime"]                     = model.get_computed_dead_time();
            diag["computedInductanceRatioK"]             = model.get_computed_inductance_ratio_k();
            diag["computedQualityFactor"]                = model.get_computed_quality_factor();
            diag["computedPrimaryResonantFrequency"]     = model.get_computed_primary_resonant_frequency();
            {
                const auto& names = model.get_per_op_name();
                const auto& mf    = model.get_per_op_mode_forward();
                const auto& mr    = model.get_per_op_mode_reverse();
                const auto& zpl   = model.get_per_op_zvs_margin_primary_lagging();
                const auto& zsl   = model.get_per_op_zvs_margin_secondary_lagging();
                const auto& zltp  = model.get_per_op_zvs_load_threshold_primary();
                const auto& zlts  = model.get_per_op_zvs_load_threshold_secondary();
                const auto& rtt   = model.get_per_op_resonant_transition_time();
                const auto& ipk   = model.get_per_op_primary_peak_current();
                const auto& spk   = model.get_per_op_secondary_peak_current();
                const auto& irms  = model.get_per_op_primary_rms_current();
                const auto& srms  = model.get_per_op_secondary_rms_current();
                const auto& mpk   = model.get_per_op_magnetizing_peak_current();
                const auto& vc1   = model.get_per_op_cr1_peak_voltage();
                const auto& vc2   = model.get_per_op_cr2_peak_voltage();
                const auto& shr   = model.get_per_op_current_sharing_ratio();
                const auto& res   = model.get_per_op_steady_state_residual();
                diag["lastModeForward"]                = mf.empty()   ? model.get_last_mode_forward()                  : mf.front();
                diag["lastModeReverse"]                = mr.empty()   ? model.get_last_mode_reverse()                  : mr.front();
                diag["lastZvsMarginPrimaryLagging"]    = zpl.empty()  ? model.get_last_zvs_margin_primary_lagging()    : zpl.front();
                diag["lastZvsMarginSecondaryLagging"]  = zsl.empty()  ? model.get_last_zvs_margin_secondary_lagging()  : zsl.front();
                diag["lastZvsLoadThresholdPrimary"]    = zltp.empty() ? model.get_last_zvs_load_threshold_primary()    : zltp.front();
                diag["lastZvsLoadThresholdSecondary"]  = zlts.empty() ? model.get_last_zvs_load_threshold_secondary()  : zlts.front();
                diag["lastResonantTransitionTime"]     = rtt.empty()  ? model.get_last_resonant_transition_time()      : rtt.front();
                diag["lastPrimaryPeakCurrent"]         = ipk.empty()  ? model.get_last_primary_peak_current()          : ipk.front();
                diag["lastSecondaryPeakCurrent"]       = spk.empty()  ? model.get_last_secondary_peak_current()        : spk.front();
                diag["lastPrimaryRmsCurrent"]          = irms.empty() ? model.get_last_primary_rms_current()           : irms.front();
                diag["lastSecondaryRmsCurrent"]        = srms.empty() ? model.get_last_secondary_rms_current()         : srms.front();
                diag["lastMagnetizingPeakCurrent"]     = mpk.empty()  ? model.get_last_magnetizing_peak_current()      : mpk.front();
                diag["lastCr1PeakVoltage"]             = vc1.empty()  ? model.get_last_cr1_peak_voltage()              : vc1.front();
                diag["lastCr2PeakVoltage"]             = vc2.empty()  ? model.get_last_cr2_peak_voltage()              : vc2.front();
                diag["lastCurrentSharingRatio"]        = shr.empty()  ? model.get_last_current_sharing_ratio()         : shr.front();
                diag["lastSteadyStateResidual"]        = res.empty()  ? model.get_last_steady_state_residual()         : res.front();
                json perOp = json::array();
                for (size_t i = 0; i < mf.size(); ++i) {
                    json row;
                    row["operatingPointName"]              = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                    row["modeForward"]                     = mf[i];
                    row["modeReverse"]                     = mr[i];
                    row["zvsMarginPrimaryLagging"]         = zpl[i];
                    row["zvsMarginSecondaryLagging"]       = zsl[i];
                    row["zvsLoadThresholdPrimary"]         = zltp[i];
                    row["zvsLoadThresholdSecondary"]       = zlts[i];
                    row["resonantTransitionTime"]          = rtt[i];
                    row["primaryPeakCurrent"]              = ipk[i];
                    row["secondaryPeakCurrent"]            = spk[i];
                    row["primaryRmsCurrent"]               = irms[i];
                    row["secondaryRmsCurrent"]             = srms[i];
                    row["magnetizingPeakCurrent"]          = mpk[i];
                    row["cr1PeakVoltage"]                  = vc1[i];
                    row["cr2PeakVoltage"]                  = vc2[i];
                    row["currentSharingRatio"]             = shr[i];
                    row["steadyStateResidual"]             = res[i];
                    perOp.push_back(row);
                }
                diag["perOp"] = perOp;
            }
            result["clllcDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_clllc_ideal_waveforms(std::string clllcInputsString) {
    try {
        json inputsJson = json::parse(clllcInputsString);
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        size_t numberOfPeriods = 2;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 5;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        OpenMagnetics::Clllc model(inputsJson);
        auto designRequirements = model.process_design_requirements();
        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) turnsRatios.push_back(tr.get_nominal().value());
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("Clllc: process_design_requirements produced no turns ratios");
        }
        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Clllc: no magnetizing inductance available");
        }
        model.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));
        auto operatingPoints = model.simulate_and_extract_operating_points(turnsRatios, magnetizingInductance);
        auto topologyWaveforms = model.simulate_and_extract_topology_waveforms(turnsRatios, magnetizingInductance, numberOfPeriods);
        json result;
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) { json j; to_json(j, op); inputs["operatingPoints"].push_back(j); }
        result["inputs"] = inputs;
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) { json j; to_json(j, tw); result["converterWaveforms"].push_back(j); }
        // Path B diagnostics: see Buck comment for rationale.
        model.process();
        // Path B field schema:
        {
            json diag;
            diag["computedPrimarySeriesInductance"]      = model.get_computed_primary_series_inductance();
            diag["computedSecondarySeriesInductance"]    = model.get_computed_secondary_series_inductance();
            diag["computedPrimaryResonantCapacitance"]   = model.get_computed_primary_resonant_capacitance();
            diag["computedSecondaryResonantCapacitance"] = model.get_computed_secondary_resonant_capacitance();
            diag["computedMagnetizingInductance"]        = model.get_computed_magnetizing_inductance();
            diag["computedTurnsRatio"]                   = model.get_computed_turns_ratio();
            diag["computedDeadTime"]                     = model.get_computed_dead_time();
            diag["computedInductanceRatioK"]             = model.get_computed_inductance_ratio_k();
            diag["computedQualityFactor"]                = model.get_computed_quality_factor();
            diag["computedPrimaryResonantFrequency"]     = model.get_computed_primary_resonant_frequency();
            {
                const auto& names = model.get_per_op_name();
                const auto& mf    = model.get_per_op_mode_forward();
                const auto& mr    = model.get_per_op_mode_reverse();
                const auto& zpl   = model.get_per_op_zvs_margin_primary_lagging();
                const auto& zsl   = model.get_per_op_zvs_margin_secondary_lagging();
                const auto& zltp  = model.get_per_op_zvs_load_threshold_primary();
                const auto& zlts  = model.get_per_op_zvs_load_threshold_secondary();
                const auto& rtt   = model.get_per_op_resonant_transition_time();
                const auto& ipk   = model.get_per_op_primary_peak_current();
                const auto& spk   = model.get_per_op_secondary_peak_current();
                const auto& irms  = model.get_per_op_primary_rms_current();
                const auto& srms  = model.get_per_op_secondary_rms_current();
                const auto& mpk   = model.get_per_op_magnetizing_peak_current();
                const auto& vc1   = model.get_per_op_cr1_peak_voltage();
                const auto& vc2   = model.get_per_op_cr2_peak_voltage();
                const auto& shr   = model.get_per_op_current_sharing_ratio();
                const auto& res   = model.get_per_op_steady_state_residual();
                diag["lastModeForward"]                = mf.empty()   ? model.get_last_mode_forward()                  : mf.front();
                diag["lastModeReverse"]                = mr.empty()   ? model.get_last_mode_reverse()                  : mr.front();
                diag["lastZvsMarginPrimaryLagging"]    = zpl.empty()  ? model.get_last_zvs_margin_primary_lagging()    : zpl.front();
                diag["lastZvsMarginSecondaryLagging"]  = zsl.empty()  ? model.get_last_zvs_margin_secondary_lagging()  : zsl.front();
                diag["lastZvsLoadThresholdPrimary"]    = zltp.empty() ? model.get_last_zvs_load_threshold_primary()    : zltp.front();
                diag["lastZvsLoadThresholdSecondary"]  = zlts.empty() ? model.get_last_zvs_load_threshold_secondary()  : zlts.front();
                diag["lastResonantTransitionTime"]     = rtt.empty()  ? model.get_last_resonant_transition_time()      : rtt.front();
                diag["lastPrimaryPeakCurrent"]         = ipk.empty()  ? model.get_last_primary_peak_current()          : ipk.front();
                diag["lastSecondaryPeakCurrent"]       = spk.empty()  ? model.get_last_secondary_peak_current()        : spk.front();
                diag["lastPrimaryRmsCurrent"]          = irms.empty() ? model.get_last_primary_rms_current()           : irms.front();
                diag["lastSecondaryRmsCurrent"]        = srms.empty() ? model.get_last_secondary_rms_current()         : srms.front();
                diag["lastMagnetizingPeakCurrent"]     = mpk.empty()  ? model.get_last_magnetizing_peak_current()      : mpk.front();
                diag["lastCr1PeakVoltage"]             = vc1.empty()  ? model.get_last_cr1_peak_voltage()              : vc1.front();
                diag["lastCr2PeakVoltage"]             = vc2.empty()  ? model.get_last_cr2_peak_voltage()              : vc2.front();
                diag["lastCurrentSharingRatio"]        = shr.empty()  ? model.get_last_current_sharing_ratio()         : shr.front();
                diag["lastSteadyStateResidual"]        = res.empty()  ? model.get_last_steady_state_residual()         : res.front();
                json perOp = json::array();
                for (size_t i = 0; i < mf.size(); ++i) {
                    json row;
                    row["operatingPointName"]              = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                    row["modeForward"]                     = mf[i];
                    row["modeReverse"]                     = mr[i];
                    row["zvsMarginPrimaryLagging"]         = zpl[i];
                    row["zvsMarginSecondaryLagging"]       = zsl[i];
                    row["zvsLoadThresholdPrimary"]         = zltp[i];
                    row["zvsLoadThresholdSecondary"]       = zlts[i];
                    row["resonantTransitionTime"]          = rtt[i];
                    row["primaryPeakCurrent"]              = ipk[i];
                    row["secondaryPeakCurrent"]            = spk[i];
                    row["primaryRmsCurrent"]               = irms[i];
                    row["secondaryRmsCurrent"]             = srms[i];
                    row["magnetizingPeakCurrent"]          = mpk[i];
                    row["cr1PeakVoltage"]                  = vc1[i];
                    row["cr2PeakVoltage"]                  = vc2[i];
                    row["currentSharingRatio"]             = shr[i];
                    row["steadyStateResidual"]             = res[i];
                    perOp.push_back(row);
                }
                diag["perOp"] = perOp;
            }
            result["clllcDiagnostics"] = diag;
        }
        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ---- Cuk (single inductanceL1, no turns ratios in simulate call) ----
std::string calculate_cuk_inputs(std::string cukInputsString) {
    try {
        json inputsJson = json::parse(cukInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::Cuk model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = model.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = model.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = model.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = model.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_ca = model.get_per_op_sized_ca();
            const auto& v_sized_cb = model.get_per_op_sized_cb();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_rhp_zero_frequency = model.get_per_op_rhp_zero_frequency();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? model.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? model.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? model.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? model.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCa"] = v_sized_ca.empty() ? model.get_last_sized_ca() : v_sized_ca.front();
            diag["sizedCb"] = v_sized_cb.empty() ? model.get_last_sized_cb() : v_sized_cb.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["rhpZeroFrequency"] = v_rhp_zero_frequency.empty() ? model.get_last_rhp_zero_frequency() : v_rhp_zero_frequency.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCa"] = v_sized_ca[i];
                row["sizedCb"] = v_sized_cb[i];
                row["sizedCo"] = v_sized_co[i];
                row["rhpZeroFrequency"] = v_rhp_zero_frequency[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["cukDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_advanced_cuk_inputs(std::string cukInputsString) {
    try {
        json inputsJson = json::parse(cukInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::AdvancedCuk model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = model.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = model.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = model.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = model.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_ca = model.get_per_op_sized_ca();
            const auto& v_sized_cb = model.get_per_op_sized_cb();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_rhp_zero_frequency = model.get_per_op_rhp_zero_frequency();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? model.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? model.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? model.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? model.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCa"] = v_sized_ca.empty() ? model.get_last_sized_ca() : v_sized_ca.front();
            diag["sizedCb"] = v_sized_cb.empty() ? model.get_last_sized_cb() : v_sized_cb.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["rhpZeroFrequency"] = v_rhp_zero_frequency.empty() ? model.get_last_rhp_zero_frequency() : v_rhp_zero_frequency.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCa"] = v_sized_ca[i];
                row["sizedCb"] = v_sized_cb[i];
                row["sizedCo"] = v_sized_co[i];
                row["rhpZeroFrequency"] = v_rhp_zero_frequency[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["cukDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_cuk_ideal_waveforms(std::string cukInputsString) {
    try {
        json inputsJson = json::parse(cukInputsString);
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        size_t numberOfPeriods = 2;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 5;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        OpenMagnetics::Cuk model(inputsJson);
        auto designRequirements = model.process_design_requirements();
        double inductanceL1;
        inductanceL1 = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(inductanceL1 > 0)) {
            throw std::runtime_error("Cuk: no inductance L1 available");
        }
        model.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));
        auto operatingPoints = model.simulate_and_extract_operating_points(inductanceL1);
        auto topologyWaveforms = model.simulate_and_extract_topology_waveforms(inductanceL1, numberOfPeriods);
        json result;
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) { json j; to_json(j, op); inputs["operatingPoints"].push_back(j); }
        result["inputs"] = inputs;
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) { json j; to_json(j, tw); result["converterWaveforms"].push_back(j); }
        // Path B diagnostics: see Buck comment for rationale.
        model.process();
        // Path B field schema:
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = model.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = model.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = model.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = model.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_ca = model.get_per_op_sized_ca();
            const auto& v_sized_cb = model.get_per_op_sized_cb();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_rhp_zero_frequency = model.get_per_op_rhp_zero_frequency();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? model.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? model.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? model.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? model.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCa"] = v_sized_ca.empty() ? model.get_last_sized_ca() : v_sized_ca.front();
            diag["sizedCb"] = v_sized_cb.empty() ? model.get_last_sized_cb() : v_sized_cb.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["rhpZeroFrequency"] = v_rhp_zero_frequency.empty() ? model.get_last_rhp_zero_frequency() : v_rhp_zero_frequency.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCa"] = v_sized_ca[i];
                row["sizedCb"] = v_sized_cb[i];
                row["sizedCo"] = v_sized_co[i];
                row["rhpZeroFrequency"] = v_rhp_zero_frequency[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["cukDiagnostics"] = diag;
        }
        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ---- FourSwitchBuckBoost (single inductance, no turns ratios) ----
std::string calculate_four_switch_buck_boost_inputs(std::string fsbbInputsString) {
    try {
        json inputsJson = json::parse(fsbbInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::FourSwitchBuckBoost model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_inductor_average_current = model.get_per_op_inductor_average_current();
            const auto& v_sized_output_capacitance = model.get_per_op_sized_output_capacitance();
            diag["inductorAverageCurrent"] = v_inductor_average_current.empty() ? model.get_last_inductor_average_current() : v_inductor_average_current.front();
            diag["sizedOutputCapacitance"] = v_sized_output_capacitance.empty() ? model.get_last_sized_output_capacitance() : v_sized_output_capacitance.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_inductor_average_current.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["inductorAverageCurrent"] = v_inductor_average_current[i];
                row["sizedOutputCapacitance"] = v_sized_output_capacitance[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["fsbbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_advanced_four_switch_buck_boost_inputs(std::string fsbbInputsString) {
    try {
        json inputsJson = json::parse(fsbbInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::AdvancedFourSwitchBuckBoost model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_inductor_average_current = model.get_per_op_inductor_average_current();
            const auto& v_sized_output_capacitance = model.get_per_op_sized_output_capacitance();
            diag["inductorAverageCurrent"] = v_inductor_average_current.empty() ? model.get_last_inductor_average_current() : v_inductor_average_current.front();
            diag["sizedOutputCapacitance"] = v_sized_output_capacitance.empty() ? model.get_last_sized_output_capacitance() : v_sized_output_capacitance.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_inductor_average_current.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["inductorAverageCurrent"] = v_inductor_average_current[i];
                row["sizedOutputCapacitance"] = v_sized_output_capacitance[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["fsbbDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_four_switch_buck_boost_ideal_waveforms(std::string fsbbInputsString) {
    try {
        json inputsJson = json::parse(fsbbInputsString);
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        size_t numberOfPeriods = 2;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 5;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        OpenMagnetics::FourSwitchBuckBoost model(inputsJson);
        auto designRequirements = model.process_design_requirements();
        double inductance;
        inductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(inductance > 0)) {
            throw std::runtime_error("FourSwitchBuckBoost: no inductance available");
        }
        model.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));
        auto operatingPoints = model.simulate_and_extract_operating_points(inductance);
        auto topologyWaveforms = model.simulate_and_extract_topology_waveforms(inductance, numberOfPeriods);
        json result;
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) { json j; to_json(j, op); inputs["operatingPoints"].push_back(j); }
        result["inputs"] = inputs;
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) { json j; to_json(j, tw); result["converterWaveforms"].push_back(j); }
        // Path B diagnostics: see Buck comment for rationale.
        model.process();
        // Path B field schema:
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_inductor_average_current = model.get_per_op_inductor_average_current();
            const auto& v_sized_output_capacitance = model.get_per_op_sized_output_capacitance();
            diag["inductorAverageCurrent"] = v_inductor_average_current.empty() ? model.get_last_inductor_average_current() : v_inductor_average_current.front();
            diag["sizedOutputCapacitance"] = v_sized_output_capacitance.empty() ? model.get_last_sized_output_capacitance() : v_sized_output_capacitance.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_inductor_average_current.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["inductorAverageCurrent"] = v_inductor_average_current[i];
                row["sizedOutputCapacitance"] = v_sized_output_capacitance[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["fsbbDiagnostics"] = diag;
        }
        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ---- Weinberg (scalar turnsRatio + magnetizingInductance) ----
std::string calculate_weinberg_inputs(std::string weinbergInputsString) {
    try {
        json inputsJson = json::parse(weinbergInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::Weinberg model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_operating_regime = model.get_per_op_operating_regime();
            const auto& v_overlap_fraction = model.get_per_op_overlap_fraction();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_energy_recovery_avg_current = model.get_per_op_energy_recovery_avg_current();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_magnetizing_ripple = model.get_per_op_magnetizing_ripple();
            const auto& v_flux_imbalance_margin = model.get_per_op_flux_imbalance_margin();
            const auto& v_rhp_zero_frequency = model.get_per_op_rhp_zero_frequency();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_output_voltage_ripple = model.get_per_op_output_voltage_ripple();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["operatingRegime"] = v_operating_regime.empty() ? model.get_last_operating_regime() : v_operating_regime.front();
            diag["overlapFraction"] = v_overlap_fraction.empty() ? model.get_last_overlap_fraction() : v_overlap_fraction.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["energyRecoveryAvgCurrent"] = v_energy_recovery_avg_current.empty() ? model.get_last_energy_recovery_avg_current() : v_energy_recovery_avg_current.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["magnetizingRipple"] = v_magnetizing_ripple.empty() ? model.get_last_magnetizing_ripple() : v_magnetizing_ripple.front();
            diag["fluxImbalanceMargin"] = v_flux_imbalance_margin.empty() ? model.get_last_flux_imbalance_margin() : v_flux_imbalance_margin.front();
            diag["rhpZeroFrequency"] = v_rhp_zero_frequency.empty() ? model.get_last_rhp_zero_frequency() : v_rhp_zero_frequency.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["outputVoltageRipple"] = v_output_voltage_ripple.empty() ? model.get_last_output_voltage_ripple() : v_output_voltage_ripple.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["operatingRegime"] = v_operating_regime[i];
                row["overlapFraction"] = v_overlap_fraction[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["energyRecoveryAvgCurrent"] = v_energy_recovery_avg_current[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["magnetizingRipple"] = v_magnetizing_ripple[i];
                row["fluxImbalanceMargin"] = v_flux_imbalance_margin[i];
                row["rhpZeroFrequency"] = v_rhp_zero_frequency[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCo"] = v_sized_co[i];
                row["outputVoltageRipple"] = v_output_voltage_ripple[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["weinbergDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_advanced_weinberg_inputs(std::string weinbergInputsString) {
    try {
        json inputsJson = json::parse(weinbergInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::AdvancedWeinberg model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_operating_regime = model.get_per_op_operating_regime();
            const auto& v_overlap_fraction = model.get_per_op_overlap_fraction();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_energy_recovery_avg_current = model.get_per_op_energy_recovery_avg_current();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_magnetizing_ripple = model.get_per_op_magnetizing_ripple();
            const auto& v_flux_imbalance_margin = model.get_per_op_flux_imbalance_margin();
            const auto& v_rhp_zero_frequency = model.get_per_op_rhp_zero_frequency();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_output_voltage_ripple = model.get_per_op_output_voltage_ripple();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["operatingRegime"] = v_operating_regime.empty() ? model.get_last_operating_regime() : v_operating_regime.front();
            diag["overlapFraction"] = v_overlap_fraction.empty() ? model.get_last_overlap_fraction() : v_overlap_fraction.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["energyRecoveryAvgCurrent"] = v_energy_recovery_avg_current.empty() ? model.get_last_energy_recovery_avg_current() : v_energy_recovery_avg_current.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["magnetizingRipple"] = v_magnetizing_ripple.empty() ? model.get_last_magnetizing_ripple() : v_magnetizing_ripple.front();
            diag["fluxImbalanceMargin"] = v_flux_imbalance_margin.empty() ? model.get_last_flux_imbalance_margin() : v_flux_imbalance_margin.front();
            diag["rhpZeroFrequency"] = v_rhp_zero_frequency.empty() ? model.get_last_rhp_zero_frequency() : v_rhp_zero_frequency.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["outputVoltageRipple"] = v_output_voltage_ripple.empty() ? model.get_last_output_voltage_ripple() : v_output_voltage_ripple.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["operatingRegime"] = v_operating_regime[i];
                row["overlapFraction"] = v_overlap_fraction[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["energyRecoveryAvgCurrent"] = v_energy_recovery_avg_current[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["magnetizingRipple"] = v_magnetizing_ripple[i];
                row["fluxImbalanceMargin"] = v_flux_imbalance_margin[i];
                row["rhpZeroFrequency"] = v_rhp_zero_frequency[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCo"] = v_sized_co[i];
                row["outputVoltageRipple"] = v_output_voltage_ripple[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["weinbergDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_weinberg_ideal_waveforms(std::string weinbergInputsString) {
    try {
        json inputsJson = json::parse(weinbergInputsString);
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        size_t numberOfPeriods = 2;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 5;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        OpenMagnetics::Weinberg model(inputsJson);
        auto designRequirements = model.process_design_requirements();
        // Weinberg simulate takes a scalar turnsRatio (single secondary)
        double turnsRatio = 0.0;
        if (!designRequirements.get_turns_ratios().empty() && designRequirements.get_turns_ratios()[0].get_nominal()) {
            turnsRatio = designRequirements.get_turns_ratios()[0].get_nominal().value();
        } else {
            throw std::runtime_error("Weinberg: process_design_requirements produced no turns ratio");
        }
        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("Weinberg: no magnetizing inductance available");
        }
        model.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));
        auto operatingPoints = model.simulate_and_extract_operating_points(turnsRatio, magnetizingInductance);
        auto topologyWaveforms = model.simulate_and_extract_topology_waveforms(turnsRatio, magnetizingInductance, numberOfPeriods);
        json result;
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) { json j; to_json(j, op); inputs["operatingPoints"].push_back(j); }
        result["inputs"] = inputs;
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) { json j; to_json(j, tw); result["converterWaveforms"].push_back(j); }
        // Path B diagnostics: see Buck comment for rationale.
        model.process();
        // Path B field schema:
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_operating_regime = model.get_per_op_operating_regime();
            const auto& v_overlap_fraction = model.get_per_op_overlap_fraction();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_energy_recovery_avg_current = model.get_per_op_energy_recovery_avg_current();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_magnetizing_ripple = model.get_per_op_magnetizing_ripple();
            const auto& v_flux_imbalance_margin = model.get_per_op_flux_imbalance_margin();
            const auto& v_rhp_zero_frequency = model.get_per_op_rhp_zero_frequency();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_output_voltage_ripple = model.get_per_op_output_voltage_ripple();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["operatingRegime"] = v_operating_regime.empty() ? model.get_last_operating_regime() : v_operating_regime.front();
            diag["overlapFraction"] = v_overlap_fraction.empty() ? model.get_last_overlap_fraction() : v_overlap_fraction.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["energyRecoveryAvgCurrent"] = v_energy_recovery_avg_current.empty() ? model.get_last_energy_recovery_avg_current() : v_energy_recovery_avg_current.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["magnetizingRipple"] = v_magnetizing_ripple.empty() ? model.get_last_magnetizing_ripple() : v_magnetizing_ripple.front();
            diag["fluxImbalanceMargin"] = v_flux_imbalance_margin.empty() ? model.get_last_flux_imbalance_margin() : v_flux_imbalance_margin.front();
            diag["rhpZeroFrequency"] = v_rhp_zero_frequency.empty() ? model.get_last_rhp_zero_frequency() : v_rhp_zero_frequency.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["outputVoltageRipple"] = v_output_voltage_ripple.empty() ? model.get_last_output_voltage_ripple() : v_output_voltage_ripple.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["operatingRegime"] = v_operating_regime[i];
                row["overlapFraction"] = v_overlap_fraction[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["energyRecoveryAvgCurrent"] = v_energy_recovery_avg_current[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["magnetizingRipple"] = v_magnetizing_ripple[i];
                row["fluxImbalanceMargin"] = v_flux_imbalance_margin[i];
                row["rhpZeroFrequency"] = v_rhp_zero_frequency[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCo"] = v_sized_co[i];
                row["outputVoltageRipple"] = v_output_voltage_ripple[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["weinbergDiagnostics"] = diag;
        }
        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ---- Zeta (single inductanceL1, no turns ratios) ----
std::string calculate_zeta_inputs(std::string zetaInputsString) {
    try {
        json inputsJson = json::parse(zetaInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::Zeta model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = model.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = model.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = model.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = model.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_cc = model.get_per_op_sized_cc();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_output_voltage_ripple = model.get_per_op_output_voltage_ripple();
            const auto& v_input_current_ripple = model.get_per_op_input_current_ripple();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? model.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? model.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? model.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? model.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCc"] = v_sized_cc.empty() ? model.get_last_sized_cc() : v_sized_cc.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["outputVoltageRipple"] = v_output_voltage_ripple.empty() ? model.get_last_output_voltage_ripple() : v_output_voltage_ripple.front();
            diag["inputCurrentRipple"] = v_input_current_ripple.empty() ? model.get_last_input_current_ripple() : v_input_current_ripple.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCc"] = v_sized_cc[i];
                row["sizedCo"] = v_sized_co[i];
                row["outputVoltageRipple"] = v_output_voltage_ripple[i];
                row["inputCurrentRipple"] = v_input_current_ripple[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["zetaDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string calculate_advanced_zeta_inputs(std::string zetaInputsString) {
    try {
        json inputsJson = json::parse(zetaInputsString);
        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        OpenMagnetics::AdvancedZeta model(inputsJson);
        auto inputs = model.process();
        json result;
        to_json(result, inputs);
        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = model.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = model.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = model.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = model.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_cc = model.get_per_op_sized_cc();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_output_voltage_ripple = model.get_per_op_output_voltage_ripple();
            const auto& v_input_current_ripple = model.get_per_op_input_current_ripple();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? model.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? model.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? model.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? model.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCc"] = v_sized_cc.empty() ? model.get_last_sized_cc() : v_sized_cc.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["outputVoltageRipple"] = v_output_voltage_ripple.empty() ? model.get_last_output_voltage_ripple() : v_output_voltage_ripple.front();
            diag["inputCurrentRipple"] = v_input_current_ripple.empty() ? model.get_last_input_current_ripple() : v_input_current_ripple.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCc"] = v_sized_cc[i];
                row["sizedCo"] = v_sized_co[i];
                row["outputVoltageRipple"] = v_output_voltage_ripple[i];
                row["inputCurrentRipple"] = v_input_current_ripple[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["zetaDiagnostics"] = diag;
        }

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_zeta_ideal_waveforms(std::string zetaInputsString) {
    try {
        json inputsJson = json::parse(zetaInputsString);
#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif
        size_t numberOfPeriods = 2;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 5;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        OpenMagnetics::Zeta model(inputsJson);
        auto designRequirements = model.process_design_requirements();
        double inductanceL1;
        inductanceL1 = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(inductanceL1 > 0)) {
            throw std::runtime_error("Zeta: no inductance L1 available");
        }
        model.set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model.set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));
        auto operatingPoints = model.simulate_and_extract_operating_points(inductanceL1);
        auto topologyWaveforms = model.simulate_and_extract_topology_waveforms(inductanceL1, numberOfPeriods);
        json result;
        json inputs;
        inputs["designRequirements"] = json();
        to_json(inputs["designRequirements"], designRequirements);
        inputs["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) { json j; to_json(j, op); inputs["operatingPoints"].push_back(j); }
        result["inputs"] = inputs;
        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) { json j; to_json(j, tw); result["converterWaveforms"].push_back(j); }
        // Path B diagnostics: see Buck comment for rationale.
        model.process();
        // Path B field schema:
        {
            json diag;
            const auto& names = model.get_per_op_name();
            const auto& v_duty_cycle = model.get_per_op_duty_cycle();
            const auto& v_conversion_ratio = model.get_per_op_conversion_ratio();
            const auto& v_coupling_cap_voltage = model.get_per_op_coupling_cap_voltage();
            const auto& v_input_inductor_average = model.get_per_op_input_inductor_average();
            const auto& v_output_inductor_average = model.get_per_op_output_inductor_average();
            const auto& v_input_inductor_ripple = model.get_per_op_input_inductor_ripple();
            const auto& v_output_inductor_ripple = model.get_per_op_output_inductor_ripple();
            const auto& v_switch_peak_voltage = model.get_per_op_switch_peak_voltage();
            const auto& v_switch_peak_current = model.get_per_op_switch_peak_current();
            const auto& v_diode_peak_reverse_voltage = model.get_per_op_diode_peak_reverse_voltage();
            const auto& v_diode_peak_current = model.get_per_op_diode_peak_current();
            const auto& v_coupling_cap_rms_current = model.get_per_op_coupling_cap_rms_current();
            const auto& v_is_ccm = model.get_per_op_is_ccm();
            const auto& v_sized_cc = model.get_per_op_sized_cc();
            const auto& v_sized_co = model.get_per_op_sized_co();
            const auto& v_output_voltage_ripple = model.get_per_op_output_voltage_ripple();
            const auto& v_input_current_ripple = model.get_per_op_input_current_ripple();
            diag["dutyCycle"] = v_duty_cycle.empty() ? model.get_last_duty_cycle() : v_duty_cycle.front();
            diag["conversionRatio"] = v_conversion_ratio.empty() ? model.get_last_conversion_ratio() : v_conversion_ratio.front();
            diag["couplingCapVoltage"] = v_coupling_cap_voltage.empty() ? model.get_last_coupling_cap_voltage() : v_coupling_cap_voltage.front();
            diag["inputInductorAverage"] = v_input_inductor_average.empty() ? model.get_last_input_inductor_average() : v_input_inductor_average.front();
            diag["outputInductorAverage"] = v_output_inductor_average.empty() ? model.get_last_output_inductor_average() : v_output_inductor_average.front();
            diag["inputInductorRipple"] = v_input_inductor_ripple.empty() ? model.get_last_input_inductor_ripple() : v_input_inductor_ripple.front();
            diag["outputInductorRipple"] = v_output_inductor_ripple.empty() ? model.get_last_output_inductor_ripple() : v_output_inductor_ripple.front();
            diag["switchPeakVoltage"] = v_switch_peak_voltage.empty() ? model.get_last_switch_peak_voltage() : v_switch_peak_voltage.front();
            diag["switchPeakCurrent"] = v_switch_peak_current.empty() ? model.get_last_switch_peak_current() : v_switch_peak_current.front();
            diag["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage.empty() ? model.get_last_diode_peak_reverse_voltage() : v_diode_peak_reverse_voltage.front();
            diag["diodePeakCurrent"] = v_diode_peak_current.empty() ? model.get_last_diode_peak_current() : v_diode_peak_current.front();
            diag["couplingCapRmsCurrent"] = v_coupling_cap_rms_current.empty() ? model.get_last_coupling_cap_rms_current() : v_coupling_cap_rms_current.front();
            diag["isCcm"] = v_is_ccm.empty() ? model.get_last_is_ccm() : (bool)v_is_ccm.front();
            diag["sizedCc"] = v_sized_cc.empty() ? model.get_last_sized_cc() : v_sized_cc.front();
            diag["sizedCo"] = v_sized_co.empty() ? model.get_last_sized_co() : v_sized_co.front();
            diag["outputVoltageRipple"] = v_output_voltage_ripple.empty() ? model.get_last_output_voltage_ripple() : v_output_voltage_ripple.front();
            diag["inputCurrentRipple"] = v_input_current_ripple.empty() ? model.get_last_input_current_ripple() : v_input_current_ripple.front();

            json perOp = json::array();
            for (size_t i = 0; i < v_duty_cycle.size(); ++i) {
                json row;
                row["operatingPointName"] = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["dutyCycle"] = v_duty_cycle[i];
                row["conversionRatio"] = v_conversion_ratio[i];
                row["couplingCapVoltage"] = v_coupling_cap_voltage[i];
                row["inputInductorAverage"] = v_input_inductor_average[i];
                row["outputInductorAverage"] = v_output_inductor_average[i];
                row["inputInductorRipple"] = v_input_inductor_ripple[i];
                row["outputInductorRipple"] = v_output_inductor_ripple[i];
                row["switchPeakVoltage"] = v_switch_peak_voltage[i];
                row["switchPeakCurrent"] = v_switch_peak_current[i];
                row["diodePeakReverseVoltage"] = v_diode_peak_reverse_voltage[i];
                row["diodePeakCurrent"] = v_diode_peak_current[i];
                row["couplingCapRmsCurrent"] = v_coupling_cap_rms_current[i];
                row["isCcm"] = (bool)v_is_ccm[i];
                row["sizedCc"] = v_sized_cc[i];
                row["sizedCo"] = v_sized_co[i];
                row["outputVoltageRipple"] = v_output_voltage_ripple[i];
                row["inputCurrentRipple"] = v_input_current_ripple[i];
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
            result["zetaDiagnostics"] = diag;
        }
        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ── SRC ─────────────────────────────────────────────────────────────────────

std::string calculate_src_inputs(std::string srcInputsString) {
    try {
        json inputsJson = json::parse(srcInputsString);

        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }

        bool isAdvanced = inputsJson.contains("desiredTurnsRatios") ||
                          inputsJson.contains("desiredResonantInductance") ||
                          inputsJson.contains("desiredResonantCapacitance");

        std::unique_ptr<OpenMagnetics::Src> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedSrc>(inputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Src>(inputsJson);
        }
        model->set_num_periods_to_extract(static_cast<int>(numberOfPeriods));

        auto designRequirements = model->process_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("SRC: process_design_requirements produced no turns ratios");
        }

        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("SRC: no magnetizing inductance available");
        }

        auto operatingPoints = model->process_operating_points(turnsRatios, magnetizingInductance);

        json result;
        json drJson;
        to_json(drJson, designRequirements);
        result["designRequirements"] = drJson;
        result["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            result["operatingPoints"].push_back(opJson);
        }

        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        json diag;
        diag["computedResonantInductance"]  = model->get_computed_resonant_inductance();
        diag["computedResonantCapacitance"] = model->get_computed_resonant_capacitance();
        diag["computedResonantFrequency"]   = model->get_computed_resonant_frequency();
        {
            const auto& names = model->get_per_op_name();
            const auto& gm    = model->get_per_op_gain_m();
            const auto& nfsw  = model->get_per_op_normalized_fsw();
            const auto& ir    = model->get_per_op_ir_peak();
            const auto& vc    = model->get_per_op_vcr_peak();
            const auto& abv   = model->get_per_op_is_above_resonance();
            diag["lastGainM"]                   = gm.empty()   ? model->get_last_gain()                : gm.front();
            diag["lastNormalizedFsw"]           = nfsw.empty() ? model->get_last_normalized_fsw()      : nfsw.front();
            diag["lastIrPeak"]                  = ir.empty()   ? model->get_last_ir_peak()             : ir.front();
            diag["lastVcrPeak"]                 = vc.empty()   ? model->get_last_vcr_peak()            : vc.front();
            diag["lastIsAboveResonance"]        = abv.empty()  ? model->get_last_is_above_resonance()  : (abv.front() != 0);
            json perOp = json::array();
            for (size_t i = 0; i < gm.size(); ++i) {
                json row;
                row["operatingPointName"]   = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["gainM"]                = gm[i];
                row["normalizedFsw"]        = nfsw[i];
                row["irPeak"]               = ir[i];
                row["vcrPeak"]              = vc[i];
                row["isAboveResonance"]     = (abv[i] != 0);
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
        }
        result["srcDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_src_ideal_waveforms(std::string srcInputsString) {
    try {
        json inputsJson = json::parse(srcInputsString);

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif

        bool isAdvanced = inputsJson.contains("desiredTurnsRatios") ||
                          inputsJson.contains("desiredResonantInductance") ||
                          inputsJson.contains("desiredResonantCapacitance");

        std::unique_ptr<OpenMagnetics::Src> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedSrc>(inputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Src>(inputsJson);
        }

        size_t numberOfPeriods = 2;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 5;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        model->set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model->set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto designRequirements = model->process_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }
        if (turnsRatios.empty()) {
            throw std::runtime_error("SRC: process_design_requirements produced no turns ratios");
        }

        double magnetizingInductance;
        magnetizingInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(magnetizingInductance > 0)) {
            throw std::runtime_error("SRC: no magnetizing inductance available");
        }

        auto topologyWaveforms = model->simulate_and_extract_topology_waveforms(
            turnsRatios, magnetizingInductance, numberOfPeriods);
        auto operatingPoints = model->simulate_and_extract_operating_points(
            turnsRatios, magnetizingInductance, numberOfPeriods);

        json result;
        json inputsJsonOut;
        inputsJsonOut["designRequirements"] = json();
        to_json(inputsJsonOut["designRequirements"], designRequirements);
        inputsJsonOut["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJsonOut["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJsonOut;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        json diag;
        diag["computedResonantInductance"]  = model->get_computed_resonant_inductance();
        diag["computedResonantCapacitance"] = model->get_computed_resonant_capacitance();
        diag["computedResonantFrequency"]   = model->get_computed_resonant_frequency();
        {
            const auto& names = model->get_per_op_name();
            const auto& gm    = model->get_per_op_gain_m();
            const auto& nfsw  = model->get_per_op_normalized_fsw();
            const auto& ir    = model->get_per_op_ir_peak();
            const auto& vc    = model->get_per_op_vcr_peak();
            const auto& abv   = model->get_per_op_is_above_resonance();
            diag["lastGainM"]                   = gm.empty()   ? model->get_last_gain()                : gm.front();
            diag["lastNormalizedFsw"]           = nfsw.empty() ? model->get_last_normalized_fsw()      : nfsw.front();
            diag["lastIrPeak"]                  = ir.empty()   ? model->get_last_ir_peak()             : ir.front();
            diag["lastVcrPeak"]                 = vc.empty()   ? model->get_last_vcr_peak()            : vc.front();
            diag["lastIsAboveResonance"]        = abv.empty()  ? model->get_last_is_above_resonance()  : (abv.front() != 0);
            json perOp = json::array();
            for (size_t i = 0; i < gm.size(); ++i) {
                json row;
                row["operatingPointName"]   = (i < names.size()) ? names[i] : ("OP " + std::to_string(i));
                row["gainM"]                = gm[i];
                row["normalizedFsw"]        = nfsw[i];
                row["irPeak"]               = ir[i];
                row["vcrPeak"]              = vc[i];
                row["isAboveResonance"]     = (abv[i] != 0);
                perOp.push_back(row);
            }
            diag["perOp"] = perOp;
        }
        result["srcDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ── Vienna ───────────────────────────────────────────────────────────────────

std::string calculate_vienna_inputs(std::string viennaInputsString) {
    try {
        json inputsJson = json::parse(viennaInputsString);

        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }

        bool isAdvanced = inputsJson.contains("desiredBoostInductance");

        std::unique_ptr<OpenMagnetics::Vienna> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedVienna>(inputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Vienna>(inputsJson);
        }
        model->set_num_periods_to_extract(static_cast<int>(numberOfPeriods));

        auto designRequirements = model->process_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

        double boostInductance;
        boostInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(boostInductance > 0)) {
            throw std::runtime_error("Vienna: no boost inductance available");
        }

        auto operatingPoints = model->process_operating_points(turnsRatios, boostInductance);

        json result;
        json drJson;
        to_json(drJson, designRequirements);
        result["designRequirements"] = drJson;
        result["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            result["operatingPoints"].push_back(opJson);
        }

        if (numberOfPeriods > 1 && result.contains("operatingPoints")) {
            repeat_operating_points_waveforms(result["operatingPoints"], numberOfPeriods);
        }

        json diag;
        diag["computedBoostInductance"]      = model->get_computed_boost_inductance();
        diag["computedModulationIndex"]      = model->get_computed_modulation_index();
        diag["computedLinePeakCurrent"]      = model->get_computed_line_peak_current();
        diag["computedSwitchVoltageStress"]  = model->get_computed_switch_voltage_stress();
        diag["lastInductorPeakCurrent"]      = model->get_last_inductor_peak_current();
        diag["lastInductorRipplePeakToPeak"] = model->get_last_inductor_ripple_peak_to_peak();
        diag["lastDutyAtPeak"]               = model->get_last_duty_at_peak();
        diag["lastSwitchVoltageStress"]      = model->get_last_switch_voltage_stress();
        diag["lastSwitchRmsCurrent"]         = model->get_last_switch_rms_current();
        diag["lastDiodeAvgCurrent"]          = model->get_last_diode_avg_current();
        diag["lastModulationIndex"]          = model->get_last_modulation_index();
        diag["lastInputPower"]               = model->get_last_input_power();
        diag["note"]                         = "Phase-1: single-phase analytical model; per-phase waveforms replicated by symmetry";
        result["viennaDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

std::string simulate_vienna_ideal_waveforms(std::string viennaInputsString) {
    try {
        json inputsJson = json::parse(viennaInputsString);

#ifndef ENABLE_NGSPICE
        throw std::runtime_error("ngspice simulation is required but ENABLE_NGSPICE was not defined at compile time");
#endif

        bool isAdvanced = inputsJson.contains("desiredBoostInductance");

        std::unique_ptr<OpenMagnetics::Vienna> model;
        if (isAdvanced) {
            model = std::make_unique<OpenMagnetics::AdvancedVienna>(inputsJson);
        } else {
            model = std::make_unique<OpenMagnetics::Vienna>(inputsJson);
        }

        size_t numberOfPeriods = 1;
        if (inputsJson.contains("numberOfPeriods")) {
            numberOfPeriods = inputsJson["numberOfPeriods"].get<size_t>();
        }
        size_t numberOfSteadyStatePeriods = 3;
        if (inputsJson.contains("numberOfSteadyStatePeriods")) {
            numberOfSteadyStatePeriods = inputsJson["numberOfSteadyStatePeriods"].get<size_t>();
        }
        model->set_num_periods_to_extract(static_cast<int>(numberOfPeriods));
        model->set_num_steady_state_periods(static_cast<int>(numberOfSteadyStatePeriods));

        auto designRequirements = model->process_design_requirements();

        std::vector<double> turnsRatios;
        for (const auto& tr : designRequirements.get_turns_ratios()) {
            if (tr.get_nominal()) {
                turnsRatios.push_back(tr.get_nominal().value());
            }
        }

        double boostInductance;
        boostInductance = OpenMagnetics::resolve_dimensional_values(designRequirements.get_magnetizing_inductance());
        if (!(boostInductance > 0)) {
            throw std::runtime_error("Vienna: no boost inductance available");
        }

        auto topologyWaveforms = model->simulate_and_extract_topology_waveforms(
            turnsRatios, boostInductance, numberOfPeriods);
        auto operatingPoints = model->simulate_and_extract_operating_points(
            turnsRatios, boostInductance, numberOfPeriods);

        json result;
        json inputsJsonOut;
        inputsJsonOut["designRequirements"] = json();
        to_json(inputsJsonOut["designRequirements"], designRequirements);
        inputsJsonOut["operatingPoints"] = json::array();
        for (const auto& op : operatingPoints) {
            json opJson;
            to_json(opJson, op);
            inputsJsonOut["operatingPoints"].push_back(opJson);
        }
        result["inputs"] = inputsJsonOut;

        result["converterWaveforms"] = json::array();
        for (const auto& tw : topologyWaveforms) {
            json cwJson;
            to_json(cwJson, tw);
            result["converterWaveforms"].push_back(cwJson);
        }

        json diag;
        diag["computedBoostInductance"]      = model->get_computed_boost_inductance();
        diag["computedModulationIndex"]      = model->get_computed_modulation_index();
        diag["computedLinePeakCurrent"]      = model->get_computed_line_peak_current();
        diag["computedSwitchVoltageStress"]  = model->get_computed_switch_voltage_stress();
        diag["note"]                         = "Phase-1: single-phase boost emulation; per-phase waveforms replicated by symmetry";
        diag["spiceMode"]                    = "singlePhaseEmulation";
        result["viennaDiagnostics"] = diag;

        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

// ── CurrentTransformer ───────────────────────────────────────────────────────

std::string process_current_transformer(std::string ctInputsString) {
    try {
        json inputsJson = json::parse(ctInputsString);

        double turnsRatio = inputsJson.value("turnsRatio", 1.0);
        double secondaryDcResistance = inputsJson.value("secondaryDcResistance", 0.0);

        OpenMagnetics::CurrentTransformer model(inputsJson);
        auto inputs = model.process(turnsRatio, secondaryDcResistance);

        json result;
        to_json(result, inputs);
        return result.dump(4);
    }
    catch (const std::exception& exc) {
        json error;
        error["error"] = std::string{exc.what()};
        return error.dump(4);
    }
}

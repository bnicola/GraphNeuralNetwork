#pragma once
#include <string>
#include <vector>

// =============================================================
//  ApplianceDataset  (v3 — per-appliance intent)
//
//  OUTPUT VECTOR: size = NUM_APPLIANCES = 5
//
//  Each output neuron encodes THREE states:
//    0.0 = not mentioned  → leave appliance state unchanged
//    0.5 = mentioned, OFF → turn this appliance off
//    1.0 = mentioned, ON  → turn this appliance on
//
//  APPLICATION LOGIC:
//    for (int i = 0; i < NUM_APPLIANCES; i++)
//    {
//        if      (output[i] > 0.75) state[i] = true;   // ON
//        else if (output[i] > 0.25) state[i] = false;  // OFF
//        // else: not mentioned — state[i] unchanged
//    }
//
//  EXAMPLES:
//    "turn the lights on and the ac off"
//      → [1.0, 0.0, 0.5, 0.0, 0.0]
//           lights=ON  ac=OFF  fan/tv/heater=untouched
//
//    "turn on the fan"
//      → [0.0, 1.0, 0.0, 0.0, 0.0]
//           fan=ON  everything else=untouched
//
//    "switch off the tv and turn on the heater"
//      → [0.0, 0.0, 0.0, 0.5, 1.0]
//           tv=OFF  heater=ON  rest=untouched
//
//    "turn off everything"
//      → [0.5, 0.5, 0.5, 0.5, 0.5]
//           all=OFF
//
//  Appliance index map:
//    0 = lights
//    1 = fan
//    2 = air conditioning
//    3 = tv
//    4 = heater
//
//  Output encoding:
//    ON  = 1.0
//    OFF = 0.5
//    --- = 0.0  (not mentioned)
// =============================================================

struct Sample
{
    std::string         sentence;
    std::vector<double> labels;   // size = NUM_APPLIANCES
};

static const int NUM_APPLIANCES = 5;
static const int OUTPUT_SIZE    = NUM_APPLIANCES;

static const std::vector<std::string> APPLIANCE_NAMES =
{
    "lights", "fan", "air conditioning", "tv", "heater"
};

static const double ON  = 1.0;
static const double OFF = 0.5;
static const double IGN = 0.0;   // not mentioned — ignore

// Helper: build label vector directly
static std::vector<double> make(double l, double f, double ac, double tv, double h)
{
    return { l, f, ac, tv, h };
}

static std::vector<Sample> buildDataset()
{
    std::vector<Sample> data;

    // ── LIGHTS ON ────────────────────────────────────────────
    data.push_back({"turn on the lights",               make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"switch on the lights",             make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"please turn on the lights",        make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"can you turn on the lights",       make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"lights on",                        make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"turn on the lamp",                 make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"switch on the lamp",               make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"i want the lights on",             make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"turn on the lighting",             make(ON,  IGN, IGN, IGN, IGN)});
    data.push_back({"could you turn the lights on",     make(ON,  IGN, IGN, IGN, IGN)});

    // ── LIGHTS OFF ───────────────────────────────────────────
    data.push_back({"turn off the lights",              make(OFF, IGN, IGN, IGN, IGN)});
    data.push_back({"switch off the lights",            make(OFF, IGN, IGN, IGN, IGN)});
    data.push_back({"lights off",                       make(OFF, IGN, IGN, IGN, IGN)});
    data.push_back({"please turn off the lights",       make(OFF, IGN, IGN, IGN, IGN)});
    data.push_back({"turn off the lamp",                make(OFF, IGN, IGN, IGN, IGN)});
    data.push_back({"can you switch off the lighting",  make(OFF, IGN, IGN, IGN, IGN)});

    // ── FAN ON ───────────────────────────────────────────────
    data.push_back({"turn on the fan",                  make(IGN, ON,  IGN, IGN, IGN)});
    data.push_back({"switch on the fan",                make(IGN, ON,  IGN, IGN, IGN)});
    data.push_back({"fan on",                           make(IGN, ON,  IGN, IGN, IGN)});
    data.push_back({"please turn on the fan",           make(IGN, ON,  IGN, IGN, IGN)});
    data.push_back({"i want the fan on",                make(IGN, ON,  IGN, IGN, IGN)});
    data.push_back({"could you turn the fan on",        make(IGN, ON,  IGN, IGN, IGN)});

    // ── FAN OFF ──────────────────────────────────────────────
    data.push_back({"turn off the fan",                 make(IGN, OFF, IGN, IGN, IGN)});
    data.push_back({"switch off the fan",               make(IGN, OFF, IGN, IGN, IGN)});
    data.push_back({"fan off",                          make(IGN, OFF, IGN, IGN, IGN)});
    data.push_back({"please turn off the fan",          make(IGN, OFF, IGN, IGN, IGN)});

    // ── AC ON ────────────────────────────────────────────────
    data.push_back({"turn on the air conditioning",         make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"switch on the air conditioning",       make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"turn on the ac",                       make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"switch on the ac",                     make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"turn on the air con",                  make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"ac on",                                make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"air con on",                           make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"please turn on the air conditioning",  make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"i want the ac on",                     make(IGN, IGN, ON,  IGN, IGN)});
    data.push_back({"could you switch the ac on",           make(IGN, IGN, ON,  IGN, IGN)});

    // ── AC OFF ───────────────────────────────────────────────
    data.push_back({"turn off the air conditioning",        make(IGN, IGN, OFF, IGN, IGN)});
    data.push_back({"switch off the ac",                    make(IGN, IGN, OFF, IGN, IGN)});
    data.push_back({"turn off the air con",                 make(IGN, IGN, OFF, IGN, IGN)});
    data.push_back({"ac off",                               make(IGN, IGN, OFF, IGN, IGN)});
    data.push_back({"please turn off the air conditioning", make(IGN, IGN, OFF, IGN, IGN)});

    // ── TV ON ────────────────────────────────────────────────
    data.push_back({"turn on the tv",                   make(IGN, IGN, IGN, ON,  IGN)});
    data.push_back({"switch on the tv",                 make(IGN, IGN, IGN, ON,  IGN)});
    data.push_back({"tv on",                            make(IGN, IGN, IGN, ON,  IGN)});
    data.push_back({"turn on the television",           make(IGN, IGN, IGN, ON,  IGN)});
    data.push_back({"please turn on the tv",            make(IGN, IGN, IGN, ON,  IGN)});
    data.push_back({"i want the tv on",                 make(IGN, IGN, IGN, ON,  IGN)});
    data.push_back({"could you turn the television on", make(IGN, IGN, IGN, ON,  IGN)});

    // ── TV OFF ───────────────────────────────────────────────
    data.push_back({"turn off the tv",                  make(IGN, IGN, IGN, OFF, IGN)});
    data.push_back({"switch off the tv",                make(IGN, IGN, IGN, OFF, IGN)});
    data.push_back({"tv off",                           make(IGN, IGN, IGN, OFF, IGN)});
    data.push_back({"turn off the television",          make(IGN, IGN, IGN, OFF, IGN)});
    data.push_back({"please turn off the tv",           make(IGN, IGN, IGN, OFF, IGN)});

    // ── HEATER ON ────────────────────────────────────────────
    data.push_back({"turn on the heater",               make(IGN, IGN, IGN, IGN, ON )});
    data.push_back({"switch on the heater",             make(IGN, IGN, IGN, IGN, ON )});
    data.push_back({"heater on",                        make(IGN, IGN, IGN, IGN, ON )});
    data.push_back({"please turn on the heater",        make(IGN, IGN, IGN, IGN, ON )});
    data.push_back({"i want the heater on",             make(IGN, IGN, IGN, IGN, ON )});
    data.push_back({"could you switch the heater on",   make(IGN, IGN, IGN, IGN, ON )});

    // ── HEATER OFF ───────────────────────────────────────────
    data.push_back({"turn off the heater",              make(IGN, IGN, IGN, IGN, OFF)});
    data.push_back({"switch off the heater",            make(IGN, IGN, IGN, IGN, OFF)});
    data.push_back({"heater off",                       make(IGN, IGN, IGN, IGN, OFF)});
    data.push_back({"please turn off the heater",       make(IGN, IGN, IGN, IGN, OFF)});

    // ── MULTI — SAME INTENT ──────────────────────────────────
    data.push_back({"turn on the lights and the fan",           make(ON,  ON,  IGN, IGN, IGN)});
    data.push_back({"lights and fan on",                        make(ON,  ON,  IGN, IGN, IGN)});
    data.push_back({"switch on the lights and fan",             make(ON,  ON,  IGN, IGN, IGN)});
    data.push_back({"turn on the lights and the ac",            make(ON,  IGN, ON,  IGN, IGN)});
    data.push_back({"lights and air conditioning on",           make(ON,  IGN, ON,  IGN, IGN)});
    data.push_back({"please turn on the lights and the ac",     make(ON,  IGN, ON,  IGN, IGN)});
    data.push_back({"turn on the fan and the ac",               make(IGN, ON,  ON,  IGN, IGN)});
    data.push_back({"fan and air con on",                       make(IGN, ON,  ON,  IGN, IGN)});
    data.push_back({"switch on the fan and air conditioning",   make(IGN, ON,  ON,  IGN, IGN)});
    data.push_back({"turn on the lights and the tv",            make(ON,  IGN, IGN, ON,  IGN)});
    data.push_back({"lights and tv on",                         make(ON,  IGN, IGN, ON,  IGN)});
    data.push_back({"turn on the heater and the lights",        make(ON,  IGN, IGN, IGN, ON )});
    data.push_back({"lights and heater on please",              make(ON,  IGN, IGN, IGN, ON )});
    data.push_back({"turn on the lights the fan and the tv",    make(ON,  ON,  IGN, ON,  IGN)});
    data.push_back({"lights fan and tv on",                     make(ON,  ON,  IGN, ON,  IGN)});
    data.push_back({"switch on the lights the ac and the tv",   make(ON,  IGN, ON,  ON,  IGN)});
    data.push_back({"turn on the fan and heater",               make(IGN, ON,  IGN, IGN, ON )});
    data.push_back({"fan and heater on",                        make(IGN, ON,  IGN, IGN, ON )});
    data.push_back({"turn on the tv and heater",                make(IGN, IGN, IGN, ON,  ON )});
    data.push_back({"tv and heater on please",                  make(IGN, IGN, IGN, ON,  ON )});
    data.push_back({"could you turn on the lights fan and ac",  make(ON,  ON,  ON,  IGN, IGN)});
    data.push_back({"please switch on the tv and the fan",      make(IGN, ON,  IGN, ON,  IGN)});
    data.push_back({"i want the lights and heater on",          make(ON,  IGN, IGN, IGN, ON )});
    data.push_back({"i want the fan and tv on",                 make(IGN, ON,  IGN, ON,  IGN)});

    data.push_back({"turn off the lights and the fan",          make(OFF, OFF, IGN, IGN, IGN)});
    data.push_back({"lights and fan off",                       make(OFF, OFF, IGN, IGN, IGN)});
    data.push_back({"switch off the lights and the ac",         make(OFF, IGN, OFF, IGN, IGN)});
    data.push_back({"turn off the fan and the ac",              make(IGN, OFF, OFF, IGN, IGN)});
    data.push_back({"fan and ac off",                           make(IGN, OFF, OFF, IGN, IGN)});
    data.push_back({"please turn off the tv and the lights",    make(OFF, IGN, IGN, OFF, IGN)});
    data.push_back({"lights and tv off",                        make(OFF, IGN, IGN, OFF, IGN)});
    data.push_back({"turn off the heater and fan",              make(IGN, OFF, IGN, IGN, OFF)});
    data.push_back({"heater and fan off",                       make(IGN, OFF, IGN, IGN, OFF)});
    data.push_back({"switch off the tv and the ac",             make(IGN, IGN, OFF, OFF, IGN)});
    data.push_back({"tv and ac off",                            make(IGN, IGN, OFF, OFF, IGN)});

    // ── MULTI — MIXED INTENT ─────────────────────────────────
    data.push_back({"turn the lights on and the ac off",            make(ON,  IGN, OFF, IGN, IGN)});
    data.push_back({"lights on and ac off",                         make(ON,  IGN, OFF, IGN, IGN)});
    data.push_back({"switch the lights on and the fan off",         make(ON,  OFF, IGN, IGN, IGN)});
    data.push_back({"lights on fan off",                            make(ON,  OFF, IGN, IGN, IGN)});
    data.push_back({"turn on the tv and turn off the lights",       make(OFF, IGN, IGN, ON,  IGN)});
    data.push_back({"tv on lights off",                             make(OFF, IGN, IGN, ON,  IGN)});
    data.push_back({"switch the heater on and the ac off",          make(IGN, IGN, OFF, IGN, ON )});
    data.push_back({"heater on ac off",                             make(IGN, IGN, OFF, IGN, ON )});
    data.push_back({"turn the fan off and the tv on",               make(IGN, OFF, IGN, ON,  IGN)});
    data.push_back({"fan off tv on",                                make(IGN, OFF, IGN, ON,  IGN)});
    data.push_back({"turn on the lights and turn off the heater",   make(ON,  IGN, IGN, IGN, OFF)});
    data.push_back({"lights on heater off",                         make(ON,  IGN, IGN, IGN, OFF)});
    data.push_back({"switch the ac on and the fan off",             make(IGN, OFF, ON,  IGN, IGN)});
    data.push_back({"ac on fan off",                                make(IGN, OFF, ON,  IGN, IGN)});
    data.push_back({"please turn the tv off and the lights on",     make(ON,  IGN, IGN, OFF, IGN)});
    data.push_back({"could you switch the heater off and fan on",   make(IGN, ON,  IGN, IGN, OFF)});
    data.push_back({"turn on the ac and switch off the tv",         make(IGN, IGN, ON,  OFF, IGN)});
    data.push_back({"ac on tv off",                                 make(IGN, IGN, ON,  OFF, IGN)});
    data.push_back({"lights on fan on ac off",                      make(ON,  ON,  OFF, IGN, IGN)});
    data.push_back({"turn the lights and fan on and the ac off",    make(ON,  ON,  OFF, IGN, IGN)});
    data.push_back({"tv on heater off lights on",                   make(ON,  IGN, IGN, ON,  OFF)});
    data.push_back({"turn on the tv and heater and turn off the fan", make(IGN, OFF, IGN, ON, ON)});

    // ── EVERYTHING ───────────────────────────────────────────
    data.push_back({"turn on everything",       make(ON,  ON,  ON,  ON,  ON )});
    data.push_back({"switch on everything",     make(ON,  ON,  ON,  ON,  ON )});
    data.push_back({"everything on",            make(ON,  ON,  ON,  ON,  ON )});
    data.push_back({"turn off everything",      make(OFF, OFF, OFF, OFF, OFF)});
    data.push_back({"switch off everything",    make(OFF, OFF, OFF, OFF, OFF)});
    data.push_back({"everything off",           make(OFF, OFF, OFF, OFF, OFF)});
    data.push_back({"please turn everything off", make(OFF, OFF, OFF, OFF, OFF)});
    data.push_back({"please turn everything on",  make(ON,  ON,  ON,  ON,  ON )});

    return data;
}

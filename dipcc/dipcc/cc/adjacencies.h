/*
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*/
#pragma once

#include "loc.h"
#include <vector>

namespace dipcc {

const std::vector<std::vector<Loc>> ADJ_A = {
    {},                                       // NONE
    {Loc::EDI, Loc::LON, Loc::LVP, Loc::WAL}, // YOR
    {Loc::YOR, Loc::LVP, Loc::CLY},           // EDI
    {Loc::YOR, Loc::WAL},                     // LON
    {Loc::YOR, Loc::EDI, Loc::WAL, Loc::CLY}, // LVP
    {},                                       // NTH
    {Loc::YOR, Loc::LON, Loc::LVP},           // WAL
    {Loc::EDI, Loc::LVP},                     // CLY
    {},                                       // NWG
    {},                                       // ENG
    {},                                       // IRI
    {},                                       // NAO
    {Loc::HOL, Loc::PIC, Loc::BUR, Loc::RUH}, // BEL
    {Loc::KIE, Loc::SWE},                     // DEN
    {},                                       // HEL
    {Loc::BEL, Loc::RUH, Loc::KIE},           // HOL
    {Loc::SWE, Loc::FIN, Loc::STP},           // NWY
    {},                                       // SKA
    {},                                       // BAR
    {Loc::PIC, Loc::GAS, Loc::PAR},           // BRE
    {},                                       // MAO
    {Loc::BEL, Loc::BRE, Loc::BUR, Loc::PAR}, // PIC
    {Loc::BEL, Loc::PIC, Loc::RUH, Loc::GAS, Loc::PAR, Loc::MAR,
     Loc::MUN},                                         // BUR
    {Loc::BEL, Loc::HOL, Loc::BUR, Loc::KIE, Loc::MUN}, // RUH
    {},                                                 // BAL
    {Loc::DEN, Loc::HOL, Loc::RUH, Loc::MUN, Loc::BER}, // KIE
    {Loc::DEN, Loc::NWY, Loc::FIN},                     // SWE
    {Loc::NWY, Loc::SWE, Loc::STP},                     // FIN
    {Loc::NWY, Loc::FIN, Loc::LVN, Loc::MOS},           // STP
    {},                                                 // STP/NC
    {Loc::BRE, Loc::BUR, Loc::PAR, Loc::SPA, Loc::MAR}, // GAS
    {Loc::BRE, Loc::PIC, Loc::BUR, Loc::GAS},           // PAR
    {Loc::TUN},                                         // NAF
    {Loc::SPA},                                         // POR
    {Loc::GAS, Loc::POR, Loc::MAR},                     // SPA
    {},                                                 // SPA/NC
    {},                                                 // SPA/SC
    {},                                                 // WES
    {Loc::BUR, Loc::GAS, Loc::SPA, Loc::PIE},           // MAR
    {Loc::BUR, Loc::RUH, Loc::KIE, Loc::BER, Loc::BOH, Loc::SIL,
     Loc::TYR},                                                   // MUN
    {Loc::KIE, Loc::MUN, Loc::PRU, Loc::SIL},                     // BER
    {},                                                           // BOT
    {Loc::STP, Loc::PRU, Loc::MOS, Loc::WAR},                     // LVN
    {Loc::BER, Loc::LVN, Loc::SIL, Loc::WAR},                     // PRU
    {},                                                           // STP/SC
    {Loc::STP, Loc::LVN, Loc::WAR, Loc::SEV, Loc::UKR},           // MOS
    {Loc::NAF},                                                   // TUN
    {},                                                           // LYO
    {},                                                           // TYS
    {Loc::MAR, Loc::TYR, Loc::TUS, Loc::VEN},                     // PIE
    {Loc::MUN, Loc::SIL, Loc::TYR, Loc::GAL, Loc::VIE},           // BOH
    {Loc::MUN, Loc::BER, Loc::PRU, Loc::BOH, Loc::WAR, Loc::GAL}, // SIL
    {Loc::MUN, Loc::PIE, Loc::BOH, Loc::VEN, Loc::VIE, Loc::TRI}, // TYR
    {Loc::LVN, Loc::PRU, Loc::MOS, Loc::SIL, Loc::UKR, Loc::GAL}, // WAR
    {Loc::MOS, Loc::UKR, Loc::ARM, Loc::RUM},                     // SEV
    {Loc::MOS, Loc::WAR, Loc::SEV, Loc::GAL, Loc::RUM},           // UKR
    {},                                                           // ION
    {Loc::PIE, Loc::ROM, Loc::VEN},                               // TUS
    {Loc::ROM, Loc::APU},                                         // NAP
    {Loc::TUS, Loc::NAP, Loc::VEN, Loc::APU},                     // ROM
    {Loc::PIE, Loc::TYR, Loc::TUS, Loc::ROM, Loc::TRI, Loc::APU}, // VEN
    {Loc::BOH, Loc::SIL, Loc::WAR, Loc::UKR, Loc::VIE, Loc::RUM,
     Loc::BUD},                                                   // GAL
    {Loc::BOH, Loc::TYR, Loc::GAL, Loc::TRI, Loc::BUD},           // VIE
    {Loc::TYR, Loc::VEN, Loc::VIE, Loc::ALB, Loc::BUD, Loc::SER}, // TRI
    {Loc::SEV, Loc::ANK, Loc::SMY, Loc::SYR},                     // ARM
    {},                                                           // BLA
    {Loc::SEV, Loc::UKR, Loc::GAL, Loc::BUD, Loc::SER, Loc::BUL}, // RUM
    {},                                                           // ADR
    {},                                                           // AEG
    {Loc::TRI, Loc::GRE, Loc::SER},                               // ALB
    {Loc::NAP, Loc::ROM, Loc::VEN},                               // APU
    {},                                                           // EAS
    {Loc::ALB, Loc::SER, Loc::BUL},                               // GRE
    {Loc::GAL, Loc::VIE, Loc::TRI, Loc::RUM, Loc::SER},           // BUD
    {Loc::TRI, Loc::RUM, Loc::ALB, Loc::GRE, Loc::BUD, Loc::BUL}, // SER
    {Loc::ARM, Loc::SMY, Loc::CON},                               // ANK
    {Loc::ARM, Loc::ANK, Loc::SYR, Loc::CON},                     // SMY
    {Loc::ARM, Loc::SMY},                                         // SYR
    {Loc::RUM, Loc::GRE, Loc::SER, Loc::CON},                     // BUL
    {},                                                           // BUL/EC
    {Loc::ANK, Loc::SMY, Loc::BUL},                               // CON
    {},                                                           // BUL/SC
};

const std::vector<std::vector<Loc>> ADJ_F = {
    {},                                       // NONE
    {Loc::EDI, Loc::LON, Loc::NTH},           // YOR
    {Loc::YOR, Loc::NTH, Loc::CLY, Loc::NWG}, // EDI
    {Loc::YOR, Loc::NTH, Loc::WAL, Loc::ENG}, // LON
    {Loc::WAL, Loc::CLY, Loc::IRI, Loc::NAO}, // LVP
    {Loc::YOR, Loc::EDI, Loc::LON, Loc::NWG, Loc::ENG, Loc::BEL, Loc::DEN,
     Loc::HEL, Loc::HOL, Loc::NWY, Loc::SKA},                     // NTH
    {Loc::LON, Loc::LVP, Loc::ENG, Loc::IRI},                     // WAL
    {Loc::EDI, Loc::LVP, Loc::NWG, Loc::NAO},                     // CLY
    {Loc::EDI, Loc::NTH, Loc::CLY, Loc::NAO, Loc::NWY, Loc::BAR}, // NWG
    {Loc::LON, Loc::NTH, Loc::WAL, Loc::IRI, Loc::BEL, Loc::BRE, Loc::MAO,
     Loc::PIC},                                                      // ENG
    {Loc::LVP, Loc::WAL, Loc::ENG, Loc::NAO, Loc::MAO},              // IRI
    {Loc::LVP, Loc::CLY, Loc::NWG, Loc::IRI, Loc::MAO},              // NAO
    {Loc::NTH, Loc::ENG, Loc::HOL, Loc::PIC},                        // BEL
    {Loc::NTH, Loc::HEL, Loc::SKA, Loc::BAL, Loc::KIE, Loc::SWE},    // DEN
    {Loc::NTH, Loc::DEN, Loc::HOL, Loc::KIE},                        // HEL
    {Loc::NTH, Loc::BEL, Loc::HEL, Loc::KIE},                        // HOL
    {Loc::NTH, Loc::NWG, Loc::SKA, Loc::BAR, Loc::SWE, Loc::STP_NC}, // NWY
    {Loc::NTH, Loc::DEN, Loc::NWY, Loc::SWE},                        // SKA
    {Loc::NWG, Loc::NWY, Loc::STP_NC},                               // BAR
    {Loc::ENG, Loc::MAO, Loc::PIC, Loc::GAS},                        // BRE
    {Loc::ENG, Loc::IRI, Loc::NAO, Loc::BRE, Loc::GAS, Loc::NAF, Loc::POR,
     Loc::SPA_NC, Loc::SPA_SC, Loc::WES}, // MAO
    {Loc::ENG, Loc::BEL, Loc::BRE},       // PIC
    {},                                   // BUR
    {},                                   // RUH
    {Loc::DEN, Loc::KIE, Loc::SWE, Loc::BER, Loc::BOT, Loc::LVN,
     Loc::PRU},                                                      // BAL
    {Loc::DEN, Loc::HEL, Loc::HOL, Loc::BAL, Loc::BER},              // KIE
    {Loc::DEN, Loc::NWY, Loc::SKA, Loc::BAL, Loc::FIN, Loc::BOT},    // SWE
    {Loc::SWE, Loc::BOT, Loc::STP_SC},                               // FIN
    {},                                                              // STP
    {Loc::NWY, Loc::BAR},                                            // STP/NC
    {Loc::BRE, Loc::MAO, Loc::SPA_NC},                               // GAS
    {},                                                              // PAR
    {Loc::MAO, Loc::WES, Loc::TUN},                                  // NAF
    {Loc::MAO, Loc::SPA_NC, Loc::SPA_SC},                            // POR
    {},                                                              // SPA
    {Loc::MAO, Loc::GAS, Loc::POR},                                  // SPA/NC
    {Loc::MAO, Loc::POR, Loc::WES, Loc::MAR, Loc::LYO},              // SPA/SC
    {Loc::MAO, Loc::NAF, Loc::SPA_SC, Loc::TUN, Loc::LYO, Loc::TYS}, // WES
    {Loc::SPA_SC, Loc::LYO, Loc::PIE},                               // MAR
    {},                                                              // MUN
    {Loc::BAL, Loc::KIE, Loc::PRU},                                  // BER
    {Loc::BAL, Loc::SWE, Loc::FIN, Loc::LVN, Loc::STP_SC},           // BOT
    {Loc::BAL, Loc::BOT, Loc::PRU, Loc::STP_SC},                     // LVN
    {Loc::BAL, Loc::BER, Loc::LVN},                                  // PRU
    {Loc::FIN, Loc::BOT, Loc::LVN},                                  // STP/SC
    {},                                                              // MOS
    {Loc::NAF, Loc::WES, Loc::TYS, Loc::ION},                        // TUN
    {Loc::SPA_SC, Loc::WES, Loc::MAR, Loc::TYS, Loc::PIE, Loc::TUS}, // LYO
    {Loc::WES, Loc::TUN, Loc::LYO, Loc::ION, Loc::TUS, Loc::NAP,
     Loc::ROM},                     // TYS
    {Loc::MAR, Loc::LYO, Loc::TUS}, // PIE
    {},                             // BOH
    {},                             // SIL
    {},                             // TYR
    {},                             // WAR
    {Loc::ARM, Loc::BLA, Loc::RUM}, // SEV
    {},                             // UKR
    {Loc::TUN, Loc::TYS, Loc::NAP, Loc::ADR, Loc::AEG, Loc::ALB, Loc::APU,
     Loc::EAS, Loc::GRE},                                            // ION
    {Loc::LYO, Loc::TYS, Loc::PIE, Loc::ROM},                        // TUS
    {Loc::TYS, Loc::ION, Loc::ROM, Loc::APU},                        // NAP
    {Loc::TYS, Loc::TUS, Loc::NAP},                                  // ROM
    {Loc::TRI, Loc::ADR, Loc::APU},                                  // VEN
    {},                                                              // GAL
    {},                                                              // VIE
    {Loc::VEN, Loc::ADR, Loc::ALB},                                  // TRI
    {Loc::SEV, Loc::BLA, Loc::ANK},                                  // ARM
    {Loc::SEV, Loc::ARM, Loc::RUM, Loc::ANK, Loc::BUL_EC, Loc::CON}, // BLA
    {Loc::SEV, Loc::BLA, Loc::BUL_EC},                               // RUM
    {Loc::ION, Loc::VEN, Loc::TRI, Loc::ALB, Loc::APU},              // ADR
    {Loc::ION, Loc::EAS, Loc::GRE, Loc::SMY, Loc::CON, Loc::BUL_SC}, // AEG
    {Loc::ION, Loc::TRI, Loc::ADR, Loc::GRE},                        // ALB
    {Loc::ION, Loc::NAP, Loc::VEN, Loc::ADR},                        // APU
    {Loc::ION, Loc::AEG, Loc::SMY, Loc::SYR},                        // EAS
    {Loc::ION, Loc::AEG, Loc::ALB, Loc::BUL_SC},                     // GRE
    {},                                                              // BUD
    {},                                                              // SER
    {Loc::ARM, Loc::BLA, Loc::CON},                                  // ANK
    {Loc::AEG, Loc::EAS, Loc::SYR, Loc::CON},                        // SMY
    {Loc::EAS, Loc::SMY},                                            // SYR
    {},                                                              // BUL
    {Loc::BLA, Loc::RUM, Loc::CON},                                  // BUL/EC
    {Loc::BLA, Loc::AEG, Loc::ANK, Loc::SMY, Loc::BUL_EC, Loc::BUL_SC}, // CON
    {Loc::AEG, Loc::GRE, Loc::CON}, // BUL/SC
};

// Same as ADJ_A except all coastal variants are included for all coastal locs.
// If ADJ_A[x] is "locs to which an at x could move", then
// ADJ_A_ALL_COASTS[x] is "locs which an army at x could support-hold or
// dislodge"
const std::vector<std::vector<Loc>> ADJ_A_ALL_COASTS = {
    {},                                                       // NONE
    {Loc::EDI, Loc::LON, Loc::LVP, Loc::WAL},                 // YOR
    {Loc::YOR, Loc::LVP, Loc::CLY},                           // EDI
    {Loc::YOR, Loc::WAL},                                     // LON
    {Loc::YOR, Loc::EDI, Loc::WAL, Loc::CLY},                 // LVP
    {},                                                       // NTH
    {Loc::YOR, Loc::LON, Loc::LVP},                           // WAL
    {Loc::EDI, Loc::LVP},                                     // CLY
    {},                                                       // NWG
    {},                                                       // ENG
    {},                                                       // IRI
    {},                                                       // NAO
    {Loc::HOL, Loc::PIC, Loc::BUR, Loc::RUH},                 // BEL
    {Loc::KIE, Loc::SWE},                                     // DEN
    {},                                                       // HEL
    {Loc::BEL, Loc::RUH, Loc::KIE},                           // HOL
    {Loc::SWE, Loc::FIN, Loc::STP, Loc::STP_NC, Loc::STP_SC}, // NWY
    {},                                                       // SKA
    {},                                                       // BAR
    {Loc::PIC, Loc::GAS, Loc::PAR},                           // BRE
    {},                                                       // MAO
    {Loc::BEL, Loc::BRE, Loc::BUR, Loc::PAR},                 // PIC
    {Loc::BEL, Loc::PIC, Loc::RUH, Loc::GAS, Loc::PAR, Loc::MAR,
     Loc::MUN},                                               // BUR
    {Loc::BEL, Loc::HOL, Loc::BUR, Loc::KIE, Loc::MUN},       // RUH
    {},                                                       // BAL
    {Loc::DEN, Loc::HOL, Loc::RUH, Loc::MUN, Loc::BER},       // KIE
    {Loc::DEN, Loc::NWY, Loc::FIN},                           // SWE
    {Loc::NWY, Loc::SWE, Loc::STP, Loc::STP_NC, Loc::STP_SC}, // FIN
    {Loc::NWY, Loc::FIN, Loc::LVN, Loc::MOS},                 // STP
    {},                                                       // STP/NC
    {Loc::BRE, Loc::BUR, Loc::PAR, Loc::SPA, Loc::SPA_NC, Loc::SPA_SC,
     Loc::MAR},                               // GAS
    {Loc::BRE, Loc::PIC, Loc::BUR, Loc::GAS}, // PAR
    {Loc::TUN},                               // NAF
    {Loc::SPA, Loc::SPA_NC, Loc::SPA_SC},     // POR
    {Loc::GAS, Loc::POR, Loc::MAR},           // SPA
    {},                                       // SPA/NC
    {},                                       // SPA/SC
    {},                                       // WES
    {Loc::BUR, Loc::GAS, Loc::SPA, Loc::SPA_NC, Loc::SPA_SC, Loc::PIE}, // MAR
    {Loc::BUR, Loc::RUH, Loc::KIE, Loc::BER, Loc::BOH, Loc::SIL,
     Loc::TYR},                                                         // MUN
    {Loc::KIE, Loc::MUN, Loc::PRU, Loc::SIL},                           // BER
    {},                                                                 // BOT
    {Loc::STP, Loc::STP_NC, Loc::STP_SC, Loc::PRU, Loc::MOS, Loc::WAR}, // LVN
    {Loc::BER, Loc::LVN, Loc::SIL, Loc::WAR},                           // PRU
    {}, // STP/SC
    {Loc::STP, Loc::STP_NC, Loc::STP_SC, Loc::LVN, Loc::WAR, Loc::SEV,
     Loc::UKR},                                                   // MOS
    {Loc::NAF},                                                   // TUN
    {},                                                           // LYO
    {},                                                           // TYS
    {Loc::MAR, Loc::TYR, Loc::TUS, Loc::VEN},                     // PIE
    {Loc::MUN, Loc::SIL, Loc::TYR, Loc::GAL, Loc::VIE},           // BOH
    {Loc::MUN, Loc::BER, Loc::PRU, Loc::BOH, Loc::WAR, Loc::GAL}, // SIL
    {Loc::MUN, Loc::PIE, Loc::BOH, Loc::VEN, Loc::VIE, Loc::TRI}, // TYR
    {Loc::LVN, Loc::PRU, Loc::MOS, Loc::SIL, Loc::UKR, Loc::GAL}, // WAR
    {Loc::MOS, Loc::UKR, Loc::ARM, Loc::RUM},                     // SEV
    {Loc::MOS, Loc::WAR, Loc::SEV, Loc::GAL, Loc::RUM},           // UKR
    {},                                                           // ION
    {Loc::PIE, Loc::ROM, Loc::VEN},                               // TUS
    {Loc::ROM, Loc::APU},                                         // NAP
    {Loc::TUS, Loc::NAP, Loc::VEN, Loc::APU},                     // ROM
    {Loc::PIE, Loc::TYR, Loc::TUS, Loc::ROM, Loc::TRI, Loc::APU}, // VEN
    {Loc::BOH, Loc::SIL, Loc::WAR, Loc::UKR, Loc::VIE, Loc::RUM,
     Loc::BUD},                                                   // GAL
    {Loc::BOH, Loc::TYR, Loc::GAL, Loc::TRI, Loc::BUD},           // VIE
    {Loc::TYR, Loc::VEN, Loc::VIE, Loc::ALB, Loc::BUD, Loc::SER}, // TRI
    {Loc::SEV, Loc::ANK, Loc::SMY, Loc::SYR},                     // ARM
    {},                                                           // BLA
    {Loc::SEV, Loc::UKR, Loc::GAL, Loc::BUD, Loc::SER, Loc::BUL, Loc::BUL_EC,
     Loc::BUL_SC},                                            // RUM
    {},                                                       // ADR
    {},                                                       // AEG
    {Loc::TRI, Loc::GRE, Loc::SER},                           // ALB
    {Loc::NAP, Loc::ROM, Loc::VEN},                           // APU
    {},                                                       // EAS
    {Loc::ALB, Loc::SER, Loc::BUL, Loc::BUL_EC, Loc::BUL_SC}, // GRE
    {Loc::GAL, Loc::VIE, Loc::TRI, Loc::RUM, Loc::SER},       // BUD
    {Loc::TRI, Loc::RUM, Loc::ALB, Loc::GRE, Loc::BUD, Loc::BUL, Loc::BUL_EC,
     Loc::BUL_SC},                                            // SER
    {Loc::ARM, Loc::SMY, Loc::CON},                           // ANK
    {Loc::ARM, Loc::ANK, Loc::SYR, Loc::CON},                 // SMY
    {Loc::ARM, Loc::SMY},                                     // SYR
    {Loc::RUM, Loc::GRE, Loc::SER, Loc::CON},                 // BUL
    {},                                                       // BUL/EC
    {Loc::ANK, Loc::SMY, Loc::BUL, Loc::BUL_EC, Loc::BUL_SC}, // CON
    {},                                                       // BUL/SC
};

// Same as ADJ_F except all coastal variants are included for all coastal locs.
// If ADJ_F[x] is "locs to which a fleet at x could move", then
// ADJ_F_ALL_COASTS[x] is "locs which a fleet at x could support-hold or
// dislodge"
const std::vector<std::vector<Loc>> ADJ_F_ALL_COASTS = {
    {},                                       // NONE
    {Loc::EDI, Loc::LON, Loc::NTH},           // YOR
    {Loc::YOR, Loc::NTH, Loc::CLY, Loc::NWG}, // EDI
    {Loc::YOR, Loc::NTH, Loc::WAL, Loc::ENG}, // LON
    {Loc::WAL, Loc::CLY, Loc::IRI, Loc::NAO}, // LVP
    {Loc::YOR, Loc::EDI, Loc::LON, Loc::NWG, Loc::ENG, Loc::BEL, Loc::DEN,
     Loc::HEL, Loc::HOL, Loc::NWY, Loc::SKA},                     // NTH
    {Loc::LON, Loc::LVP, Loc::ENG, Loc::IRI},                     // WAL
    {Loc::EDI, Loc::LVP, Loc::NWG, Loc::NAO},                     // CLY
    {Loc::EDI, Loc::NTH, Loc::CLY, Loc::NAO, Loc::NWY, Loc::BAR}, // NWG
    {Loc::LON, Loc::NTH, Loc::WAL, Loc::IRI, Loc::BEL, Loc::BRE, Loc::MAO,
     Loc::PIC},                                                   // ENG
    {Loc::LVP, Loc::WAL, Loc::ENG, Loc::NAO, Loc::MAO},           // IRI
    {Loc::LVP, Loc::CLY, Loc::NWG, Loc::IRI, Loc::MAO},           // NAO
    {Loc::NTH, Loc::ENG, Loc::HOL, Loc::PIC},                     // BEL
    {Loc::NTH, Loc::HEL, Loc::SKA, Loc::BAL, Loc::KIE, Loc::SWE}, // DEN
    {Loc::NTH, Loc::DEN, Loc::HOL, Loc::KIE},                     // HEL
    {Loc::NTH, Loc::BEL, Loc::HEL, Loc::KIE},                     // HOL
    {Loc::NTH, Loc::NWG, Loc::SKA, Loc::BAR, Loc::SWE, Loc::STP, Loc::STP_NC,
     Loc::STP_SC},                                            // NWY
    {Loc::NTH, Loc::DEN, Loc::NWY, Loc::SWE},                 // SKA
    {Loc::NWG, Loc::NWY, Loc::STP, Loc::STP_NC, Loc::STP_SC}, // BAR
    {Loc::ENG, Loc::MAO, Loc::PIC, Loc::GAS},                 // BRE
    {Loc::ENG, Loc::IRI, Loc::NAO, Loc::BRE, Loc::GAS, Loc::NAF, Loc::POR,
     Loc::SPA, Loc::SPA_NC, Loc::SPA_SC, Loc::WES}, // MAO
    {Loc::ENG, Loc::BEL, Loc::BRE},                 // PIC
    {},                                             // BUR
    {},                                             // RUH
    {Loc::DEN, Loc::KIE, Loc::SWE, Loc::BER, Loc::BOT, Loc::LVN,
     Loc::PRU},                                                   // BAL
    {Loc::DEN, Loc::HEL, Loc::HOL, Loc::BAL, Loc::BER},           // KIE
    {Loc::DEN, Loc::NWY, Loc::SKA, Loc::BAL, Loc::FIN, Loc::BOT}, // SWE
    {Loc::SWE, Loc::BOT, Loc::STP, Loc::STP_NC, Loc::STP_SC},     // FIN
    {},                                                           // STP
    {Loc::NWY, Loc::BAR},                                         // STP/NC
    {Loc::BRE, Loc::MAO, Loc::SPA, Loc::SPA_NC, Loc::SPA_SC},     // GAS
    {},                                                           // PAR
    {Loc::MAO, Loc::WES, Loc::TUN},                               // NAF
    {Loc::MAO, Loc::SPA, Loc::SPA_NC, Loc::SPA_SC},               // POR
    {},                                                           // SPA
    {Loc::MAO, Loc::GAS, Loc::POR},                               // SPA/NC
    {Loc::MAO, Loc::POR, Loc::WES, Loc::MAR, Loc::LYO},           // SPA/SC
    {Loc::MAO, Loc::NAF, Loc::SPA, Loc::SPA_NC, Loc::SPA_SC, Loc::TUN, Loc::LYO,
     Loc::TYS},                                               // WES
    {Loc::SPA, Loc::SPA_NC, Loc::SPA_SC, Loc::LYO, Loc::PIE}, // MAR
    {},                                                       // MUN
    {Loc::BAL, Loc::KIE, Loc::PRU},                           // BER
    {Loc::BAL, Loc::SWE, Loc::FIN, Loc::LVN, Loc::STP, Loc::STP_NC,
     Loc::STP_SC},                                                      // BOT
    {Loc::BAL, Loc::BOT, Loc::PRU, Loc::STP, Loc::STP_NC, Loc::STP_SC}, // LVN
    {Loc::BAL, Loc::BER, Loc::LVN},                                     // PRU
    {Loc::FIN, Loc::BOT, Loc::LVN},           // STP/SC
    {},                                       // MOS
    {Loc::NAF, Loc::WES, Loc::TYS, Loc::ION}, // TUN
    {Loc::SPA, Loc::SPA_NC, Loc::SPA_SC, Loc::WES, Loc::MAR, Loc::TYS, Loc::PIE,
     Loc::TUS}, // LYO
    {Loc::WES, Loc::TUN, Loc::LYO, Loc::ION, Loc::TUS, Loc::NAP,
     Loc::ROM},                     // TYS
    {Loc::MAR, Loc::LYO, Loc::TUS}, // PIE
    {},                             // BOH
    {},                             // SIL
    {},                             // TYR
    {},                             // WAR
    {Loc::ARM, Loc::BLA, Loc::RUM}, // SEV
    {},                             // UKR
    {Loc::TUN, Loc::TYS, Loc::NAP, Loc::ADR, Loc::AEG, Loc::ALB, Loc::APU,
     Loc::EAS, Loc::GRE},                     // ION
    {Loc::LYO, Loc::TYS, Loc::PIE, Loc::ROM}, // TUS
    {Loc::TYS, Loc::ION, Loc::ROM, Loc::APU}, // NAP
    {Loc::TYS, Loc::TUS, Loc::NAP},           // ROM
    {Loc::TRI, Loc::ADR, Loc::APU},           // VEN
    {},                                       // GAL
    {},                                       // VIE
    {Loc::VEN, Loc::ADR, Loc::ALB},           // TRI
    {Loc::SEV, Loc::BLA, Loc::ANK},           // ARM
    {Loc::SEV, Loc::ARM, Loc::RUM, Loc::ANK, Loc::BUL, Loc::BUL_SC, Loc::BUL_EC,
     Loc::CON},                                               // BLA
    {Loc::SEV, Loc::BLA, Loc::BUL, Loc::BUL_SC, Loc::BUL_EC}, // RUM
    {Loc::ION, Loc::VEN, Loc::TRI, Loc::ALB, Loc::APU},       // ADR
    {Loc::ION, Loc::EAS, Loc::GRE, Loc::SMY, Loc::CON, Loc::BUL, Loc::BUL_SC,
     Loc::BUL_EC},                                                      // AEG
    {Loc::ION, Loc::TRI, Loc::ADR, Loc::GRE},                           // ALB
    {Loc::ION, Loc::NAP, Loc::VEN, Loc::ADR},                           // APU
    {Loc::ION, Loc::AEG, Loc::SMY, Loc::SYR},                           // EAS
    {Loc::ION, Loc::AEG, Loc::ALB, Loc::BUL, Loc::BUL_SC, Loc::BUL_EC}, // GRE
    {},                                                                 // BUD
    {},                                                                 // SER
    {Loc::ARM, Loc::BLA, Loc::CON},                                     // ANK
    {Loc::AEG, Loc::EAS, Loc::SYR, Loc::CON},                           // SMY
    {Loc::EAS, Loc::SMY},                                               // SYR
    {},                                                                 // BUL
    {Loc::BLA, Loc::RUM, Loc::CON}, // BUL/EC
    {Loc::BLA, Loc::AEG, Loc::ANK, Loc::SMY, Loc::BUL, Loc::BUL_EC,
     Loc::BUL_SC},                  // CON
    {Loc::AEG, Loc::GRE, Loc::CON}, // BUL/SC
};

} // namespace dipcc

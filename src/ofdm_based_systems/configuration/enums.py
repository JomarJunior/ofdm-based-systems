from enum import Enum


class ConstellationType(str, Enum):
    QAM = "QAM"
    PSK = "PSK"

    def __str__(self):
        return self.value


class PrefixType(str, Enum):
    CYCLIC = "CYCLIC"
    ZERO = "ZERO"
    NONE = "NONE"

    def __str__(self):
        return self.value


class EqualizationMethod(str, Enum):
    ZF = "ZF"
    MMSE = "MMSE"
    NONE = "NONE"

    def __str__(self):
        return self.value


class ModulationType(str, Enum):
    OFDM = "OFDM"
    SC_OFDM = "SC-OFDM"

    def __str__(self):
        return self.value


class ChannelType(str, Enum):
    FLAT = "FLAT"
    CUSTOM = "CUSTOM"

    def __str__(self):
        return self.value


class NoiseType(str, Enum):
    AWGN = "AWGN"
    NONE = "NONE"

    def __str__(self):
        return self.value


class PowerAllocationType(str, Enum):
    UNIFORM = "UNIFORM"
    WATERFILLING = "WATERFILLING"

    def __str__(self):
        return self.value


class AdaptiveModulationMode(str, Enum):
    FIXED = "FIXED"
    CAPACITY_BASED = "CAPACITY_BASED"

    def __str__(self):
        return self.value

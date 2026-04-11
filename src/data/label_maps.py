from __future__ import annotations

NSL_NORMAL = {"normal"}

NSL_DOS = {
    "back", "land", "neptune", "pod", "smurf", "teardrop",
    "mailbomb", "apache2", "processtable", "udpstorm",
}

NSL_PROBE = {
    "ipsweep", "nmap", "portsweep", "satan", "mscan", "saint",
}

NSL_R2L = {
    "ftp_write", "guess_passwd", "imap", "multihop", "phf",
    "spy", "warezclient", "warezmaster", "sendmail", "named",
    "snmpgetattack", "snmpguess", "xlock", "xsnoop", "worm",
}

NSL_U2R = {
    "buffer_overflow", "loadmodule", "perl", "rootkit",
    "httptunnel", "ps", "sqlattack", "xterm",
}


def map_nsl_label_to_5class(label: str) -> str:
    label = str(label).strip().lower()
    if label in NSL_NORMAL:
        return "Normal"
    if label in NSL_DOS:
        return "DoS"
    if label in NSL_PROBE:
        return "Probe"
    if label in NSL_R2L:
        return "R2L"
    if label in NSL_U2R:
        return "U2R"
    return "Unknown"

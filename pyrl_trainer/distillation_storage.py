"""Bounded producer storage provenance, independent of world and trainer RAM."""
from dataclasses import dataclass

from .distillation_transport import (
    MAX_PRODUCER_STORAGE_BYTES, DistillationContractError, exact_keys, hex_identity, integer,
)


@dataclass(frozen=True)
class ProducerStoragePolicy:
    """Immutable admitted byte/count/lease ceilings across producer datasets."""

    max_bytes: int
    max_chunks: int
    max_datasets: int
    max_leases: int
    max_pinned_chunks: int
    max_pinned_bytes: int
    max_lease_ms: int
    disk_reserve_bytes: int

    @classmethod
    def from_object(cls, value):
        """Validate a complete policy before accepting any source cursor or chunk."""
        exact_keys(value, "schemaVersion scope maxBytes maxChunks maxDatasets maxLeases "
                   "maxPinnedChunks maxPinnedBytes maxLeaseMs diskReserveBytes")
        integer(value["schemaVersion"], 1, 1)
        if value["scope"] != "producer-global":
            raise DistillationContractError("unsupported producer storage scope")
        def word(name, low, high):
            """Read one admitted exact unsigned byte word."""
            return integer(int(hex_identity(value[name], 16, True), 16), low, high)
        return cls(
            word("maxBytes", 1, MAX_PRODUCER_STORAGE_BYTES),
            integer(value["maxChunks"], 1, 65536),
            integer(value["maxDatasets"], 1, 256),
            integer(value["maxLeases"], 1, 64),
            integer(value["maxPinnedChunks"], 1, 1024),
            word("maxPinnedBytes", 1, 512 * 1024**2),
            integer(value["maxLeaseMs"], 1, 300000),
            word("diskReserveBytes", 1024**3, (1 << 53) - 1),
        )

    def to_object(self):
        """Persist exact observed policy; this is a ceiling, not a disk reservation."""
        return {"schemaVersion": 1, "scope": "producer-global",
                "maxBytes": f"{self.max_bytes:016x}", "maxChunks": self.max_chunks,
                "maxDatasets": self.max_datasets, "maxLeases": self.max_leases,
                "maxPinnedChunks": self.max_pinned_chunks,
                "maxPinnedBytes": f"{self.max_pinned_bytes:016x}",
                "maxLeaseMs": self.max_lease_ms,
                "diskReserveBytes": f"{self.disk_reserve_bytes:016x}"}

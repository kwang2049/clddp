from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import logging
import os
from typing import List, Optional, Set, Type
from clddp.utils import get_commit_hash, parse_cli
from abc import ABC, abstractmethod


@dataclass
class AutoRunNameArgumentsMixIn:
    run_name: Optional[str] = None

    @property
    def escaped_args(self) -> Set[str]:
        """Escaped items for building the run_name."""
        return set()

    def get_arguments_from(self) -> List[type]:
        """Return the classes which the auto_run_name will get the arguments from."""
        return [self.__class__]

    @property
    def auto_run_name(self) -> str:
        arguments = [
            k
            for from_class in self.get_arguments_from()
            for k in from_class.__annotations__.keys()
            if k not in self.escaped_args
        ]
        items = [f"{argument}_{getattr(self, argument)}" for argument in arguments]
        return "/".join(items + [f"git_hash_{get_commit_hash()}"])

    def __post_init__(self) -> None:
        if self.run_name is None:
            self.run_name = self.auto_run_name
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # Needed as a MixIn


@dataclass
class DumpableArgumentsMixIn(ABC):
    output_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.output_dir is None:
            self.output_dir = self.build_output_dir()
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # Needed as a MixIn

    @abstractmethod
    def build_output_dir(self) -> str:
        pass

    def dump_arguments(self, fname: str = "arguments.json") -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        fargs = os.path.join(self.output_dir, fname)
        with open(fargs, "w") as f:
            json.dump(asdict(self), f, indent=4)
        logging.info(f"Dumped arguments to {fargs}")

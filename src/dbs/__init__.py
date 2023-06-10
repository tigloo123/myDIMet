import logging
from pydantic.dataclasses import dataclass
from pydantic import validator

logger = logging.getLogger(__name__)


@dataclass
class DBConnection:
    def connect(self):
        ...


@dataclass
class MySQLConnection(DBConnection):
    host: str
    port: int
    user: str
    password: str

    def connect(self) -> None:
        print(f"MySQL connecting to {self.host}")


@dataclass
class PostgreSQLConnection(DBConnection):
    host: str
    port: int
    user: str
    password: str
    database: str

    def connect(self) -> None:
        print(f"PostgreSQL connecting to {self.host} via port {self.port}")
    @validator("port")
    def validate_port(cls, port: int) -> int:
        if port < 0:
            raise ValueError(f"'port' can't be less than 0, got: {port}")
        return port

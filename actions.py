from enum import Enum, IntEnum
from typing import Union, NamedTuple, Tuple

QueryType = IntEnum("QueryType", "And Or Singleton Cascade", module=__name__)


class Query(NamedTuple):
    """ Information about the query to be executed """
    endpoints: Union[str, Tuple[str, str]]
    type: QueryType


class ExhaustedQueriesException(Exception):
    """ Raise this exception when there are no queries left """
    pass
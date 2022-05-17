macro(_ofa_find _list _value _ret)
  list(FIND ${_list} "${_value}" _found)
  if(_found EQUAL -1)
    set(${_ret} FALSE)
  else()
    set(${_ret} TRUE)
  endif()
endmacro(_ofa_find)
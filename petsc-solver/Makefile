.PHONY: all default clean

all: ghiep.out hils.out ghep_reduced.out
default: all

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ghiep.out: ghiep.o
	${CLINKER} -o ghiep.out ghiep.o ${SLEPC_EPS_LIB}
	${RM} ghiep.o

ghep_reduced.out: ghep_reduced.o
	${CLINKER} -o ghep_reduced.out ghep_reduced.o ${SLEPC_EPS_LIB}
	${RM} ghep_reduced.o

hils.out: hils.o
	${CLINKER} -o hils.out hils.o ${SLEPC_EPS_LIB}
	${RM} hils.o

clean::
	${RM} ghiep.o ghep_reduced.o hils.o ghiep.out ghep_reduced.out hils.out

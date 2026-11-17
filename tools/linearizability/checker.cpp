// The checker is header-only (it's templated on the spec). Instantiating it
// for the built-in specs here means the tools library compiles the whole
// search even in a build without tests.
#include "linearizability/checker.hpp"

#include "linearizability/spec.hpp"

namespace lockfree::lin {

template CheckResult check<FifoQueueSpec>(const History&, const CheckOptions&);
template CheckResult check<LifoStackSpec>(const History&, const CheckOptions&);

} // namespace lockfree::lin

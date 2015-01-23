#ifndef STORM_LOGIC_ATOMICLABELFORMULA_H_
#define STORM_LOGIC_ATOMICLABELFORMULA_H_

#include <string>

#include "src/properties/logic/StateFormula.h"

namespace storm {
    namespace logic {
        class AtomicLabelFormula : public StateFormula {
        public:
            virtual ~AtomicLabelFormula() {
                // Intentionally left empty.
            }
            
            virtual bool isAtomicLabelFormula() const override;

            std::string const& getLabel() const;
            
            virtual std::ostream& writeToStream(std::ostream& out) const override;
            
        private:
            std::string label;
        };
    }
}

#endif /* STORM_LOGIC_ATOMICLABELFORMULA_H_ */
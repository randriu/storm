/*
 * ConstantExpression.cpp
 *
 *  Created on: 10.06.2013
 *      Author: Christian Dehnert
 */

#include "ConstantExpression.h"

namespace storm {
    namespace ir {
        namespace expressions {
            
            ConstantExpression::ConstantExpression(ReturnType type, std::string const& constantName) : BaseExpression(type), constantName(constantName) {
                // Nothing to do here.
            }
            
            std::string const& ConstantExpression::getConstantName() const {
                return constantName;
            }
            
            std::string ConstantExpression::toString() const {
                return constantName;
            }
            
        } // namespace expressions
    } // namespace ir
} // namespace storm
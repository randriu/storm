#include "storm/generator/TransientVariableInformation.h"

#include "storm/storage/jani/Model.h"

#include "storm/storage/jani/Automaton.h"
#include "storm/storage/jani/eliminator/ArrayEliminator.h"
#include "storm/storage/jani/AutomatonComposition.h"
#include "storm/storage/jani/ParallelComposition.h"
#include "storm/storage/expressions/ExpressionManager.h"

#include "storm/utility/macros.h"
#include "storm/exceptions/InvalidArgumentException.h"
#include "storm/exceptions/WrongFormatException.h"
#include "storm/exceptions/OutOfRangeException.h"
#include "JaniNextStateGenerator.h"


#include <cmath>

namespace storm {
    namespace generator {
        
        template <>
        TransientVariableData<storm::RationalFunction>::TransientVariableData(storm::expressions::Variable const& variable,  boost::optional<storm::RationalFunction> const& lowerBound, boost::optional<storm::RationalFunction> const& upperBound, storm::RationalFunction const& defaultValue, bool global) : variable(variable), lowerBound(lowerBound), upperBound(upperBound), defaultValue(defaultValue) {
            // There is no '<=' for rational functions. Therefore, do not check the bounds for this ValueType
        }
        
        template <typename VariableType>
        TransientVariableData<VariableType>::TransientVariableData(storm::expressions::Variable const& variable,  boost::optional<VariableType> const& lowerBound, boost::optional<VariableType> const& upperBound, VariableType const& defaultValue, bool global) : variable(variable), lowerBound(lowerBound), upperBound(upperBound), defaultValue(defaultValue) {
            STORM_LOG_THROW(!lowerBound.is_initialized() || lowerBound.get() <= defaultValue, storm::exceptions::OutOfRangeException, "The default value for transient variable " << variable.getName() << " is smaller than its lower bound.");
            STORM_LOG_THROW(!upperBound.is_initialized() || defaultValue <= upperBound.get(), storm::exceptions::OutOfRangeException, "The default value for transient variable " << variable.getName() << " is higher than its upper bound.");
        }
        
        template <typename VariableType>
        TransientVariableData<VariableType>::TransientVariableData(storm::expressions::Variable const& variable, VariableType const& defaultValue, bool global) : variable(variable), defaultValue(defaultValue) {
            // Intentionally left empty.
        }
        
        template <typename ValueType>
        TransientVariableInformation<ValueType>::TransientVariableInformation(storm::jani::Model const& model, std::vector<std::reference_wrapper<storm::jani::Automaton const>> const& parallelAutomata) {
            
            createVariablesForVariableSet(model.getGlobalVariables(), true);
            
            for (auto const& automatonRef : parallelAutomata) {
                createVariablesForAutomaton(automatonRef.get());
            }
            
            sortVariables();
        }
        
        template <typename ValueType>
        void TransientVariableInformation<ValueType>::registerArrayVariableReplacements(storm::jani::ArrayEliminatorData const& arrayEliminatorData) {
            arrayVariableToElementInformations.clear();
            // Find for each replaced array variable the corresponding references in this variable information
            for (auto const& arrayVariable : arrayEliminatorData.eliminatedArrayVariables) {
                if (arrayVariable->isTransient()) {
                    STORM_LOG_ASSERT(arrayEliminatorData.replacements.count(arrayVariable->getExpressionVariable()) > 0, "No replacement for array variable.");
                    auto const& replacements = arrayEliminatorData.replacements.find(arrayVariable->getExpressionVariable())->second;
                    std::vector<uint64_t> varInfoIndices;
                    for (auto const& replacedVar : replacements) {
                        if (replacedVar->getExpressionVariable().hasIntegerType()) {
                            uint64_t index = 0;
                            for (auto const& intInfo : integerVariableInformation) {
                                if (intInfo.variable == replacedVar->getExpressionVariable()) {
                                    varInfoIndices.push_back(index);
                                    break;
                                }
                                ++index;
                            }
                            STORM_LOG_ASSERT(!varInfoIndices.empty() && varInfoIndices.back() == index, "Could not find a basic variable for replacement of array variable " << replacedVar->getExpressionVariable().getName() << " .");
                        } else if (replacedVar->getExpressionVariable().hasBooleanType()) {
                            uint64_t index = 0;
                            for (auto const& boolInfo : booleanVariableInformation) {
                                if (boolInfo.variable == replacedVar->getExpressionVariable()) {
                                    varInfoIndices.push_back(index);
                                    break;
                                }
                                ++index;
                            }
                            STORM_LOG_ASSERT(!varInfoIndices.empty() && varInfoIndices.back() == index, "Could not find a basic variable for replacement of array variable " << replacedVar->getExpressionVariable().getName() << " .");
                        } else if (replacedVar->getExpressionVariable().hasRationalType()) {
                            uint64_t index = 0;
                            for (auto const& rationalInfo : rationalVariableInformation) {
                                if (rationalInfo.variable == replacedVar->getExpressionVariable()) {
                                    varInfoIndices.push_back(index);
                                    break;
                                }
                                ++index;
                            }
                            STORM_LOG_ASSERT(!varInfoIndices.empty() && varInfoIndices.back() == index, "Could not find a basic variable for replacement of array variable " << replacedVar->getExpressionVariable().getName() << " .");
                        } else {
                            STORM_LOG_ASSERT(false, "Unhandled type of base variable.");
                        }
                    }
                    assert (false);
                    // TODO: Implement this properly
//                    this->arrayVariableToElementInformations.emplace(arrayVariable->getExpressionVariable(), std::move(varInfoIndices));
                }
            }
        }
        
        template <typename ValueType>
        TransientVariableData<bool> const& TransientVariableInformation<ValueType>::getBooleanArrayVariableReplacement(storm::expressions::Variable const& arrayVariable, std::vector<uint64_t>& arrayIndex) const {
            auto arrayInfoPtr = arrayVariableToElementInformations.find(arrayVariable);
            assert (arrayInfoPtr != arrayVariableToElementInformations.end());
            ArrayInformation arrayInfo = arrayInfoPtr->second;

            auto i = 0;
            while (arrayInfo.arrayIndexMapping.size() > 0) {
                assert (arrayInfo.indexMapping.size() == 0);
                assert (i < arrayIndex.size() - 1);
                STORM_LOG_THROW(arrayIndex.at(i) < arrayInfo.size, storm::exceptions::WrongFormatException, "Array access at array " << arrayVariable.getName() << " evaluates to array index " << arrayIndex.at(i) << " which is out of bounds as the array size is " << arrayInfo.size);
                i++;
                arrayInfo = arrayInfo.arrayIndexMapping.at(arrayIndex[i]);
            }
            assert (i < arrayIndex.size());

            return booleanVariableInformation[arrayInfo.indexMapping.at(arrayIndex[i])];
        }
        
        template <typename ValueType>
        TransientVariableData<int64_t> const& TransientVariableInformation<ValueType>::getIntegerArrayVariableReplacement(storm::expressions::Variable const& arrayVariable, std::vector<uint64_t>& arrayIndex) const {
            auto arrayInfoPtr = arrayVariableToElementInformations.find(arrayVariable);
            assert (arrayInfoPtr != arrayVariableToElementInformations.end());
            ArrayInformation arrayInfo = arrayInfoPtr->second;

            auto i = 0;
            while (arrayInfo.arrayIndexMapping.size() > 0) {
                assert (arrayInfo.indexMapping.size() == 0);
                assert (i < arrayIndex.size() - 1);
                STORM_LOG_THROW(arrayIndex.at(i) < arrayInfo.size, storm::exceptions::WrongFormatException, "Array access at array " << arrayVariable.getName() << " evaluates to array index " << arrayIndex.at(i) << " which is out of bounds as the array size is " << arrayInfo.size);
                i++;
                arrayInfo = arrayInfo.arrayIndexMapping.at(arrayIndex[i]);
            }
            assert (i < arrayIndex.size());

            return integerVariableInformation[arrayInfo.indexMapping.at(arrayIndex[i])];
        }
        
        template <typename ValueType>
        TransientVariableData<ValueType> const& TransientVariableInformation<ValueType>::getRationalArrayVariableReplacement(storm::expressions::Variable const& arrayVariable, std::vector<uint64_t>& arrayIndex) const {
            auto arrayInfoPtr = arrayVariableToElementInformations.find(arrayVariable);
            assert (arrayInfoPtr != arrayVariableToElementInformations.end());
            ArrayInformation arrayInfo = arrayInfoPtr->second;

            auto i = 0;
            while (arrayInfo.arrayIndexMapping.size() > 0) {
                assert (arrayInfo.indexMapping.size() == 0);
                assert (i < arrayIndex.size() - 1);
                STORM_LOG_THROW(arrayIndex.at(i) < arrayInfo.size, storm::exceptions::WrongFormatException, "Array access at array " << arrayVariable.getName() << " evaluates to array index " << arrayIndex.at(i) << " which is out of bounds as the array size is " << arrayInfo.size);
                i++;
                arrayInfo = arrayInfo.arrayIndexMapping.at(arrayIndex[i]);
            }
            assert (i < arrayIndex.size());

            return rationalVariableInformation[arrayInfo.indexMapping.at(arrayIndex[i])];
        }
        
        template <typename ValueType>
        void TransientVariableInformation<ValueType>::createVariablesForAutomaton(storm::jani::Automaton const& automaton) {
            createVariablesForVariableSet(automaton.getVariables(), false);
        }
        
        template <typename ValueType>
        void TransientVariableInformation<ValueType>::createVariablesForVariableSet(storm::jani::VariableSet const& variableSet, bool global) {
            for (auto const& variable : variableSet.getBooleanVariables()) {
                if (variable.isTransient()) {
                    booleanVariableInformation.emplace_back(variable.getExpressionVariable(), variable.getInitExpression().evaluateAsBool(), global);
                }
            }
            for (auto const& variable : variableSet.getBoundedIntegerVariables()) {
                if (variable.isTransient()) {
                    boost::optional<int64_t> lowerBound;
                    boost::optional<int64_t> upperBound;
                    if (variable.hasLowerBound()) {
                        lowerBound = variable.getLowerBound().evaluateAsInt();
                    }
                    if (variable.hasUpperBound()) {
                        upperBound = variable.getUpperBound().evaluateAsInt();
                    }
                    integerVariableInformation.emplace_back(variable.getExpressionVariable(), lowerBound, upperBound, variable.getInitExpression().evaluateAsInt(), global);
                }
            }
            for (auto const& variable : variableSet.getUnboundedIntegerVariables()) {
                if (variable.isTransient()) {
                    integerVariableInformation.emplace_back(variable.getExpressionVariable(), variable.getInitExpression().evaluateAsInt(), global);
                }
            }
            for (auto const& variable : variableSet.getRealVariables()) {
                if (variable.isTransient()) {
                    rationalVariableInformation.emplace_back(variable.getExpressionVariable(), storm::utility::convertNumber<ValueType>(variable.getInitExpression().evaluateAsRational()), global);
                }
            }
        }
        
        template <typename ValueType>
        void TransientVariableInformation<ValueType>::sortVariables() {
            // Sort the variables so we can make some assumptions when iterating over them (in the next-state generators).
            std::sort(booleanVariableInformation.begin(), booleanVariableInformation.end(), [] (TransientVariableData<bool> const& a, TransientVariableData<bool> const& b) { return a.variable < b.variable; } );
            std::sort(integerVariableInformation.begin(), integerVariableInformation.end(), [] (TransientVariableData<int64_t> const& a, TransientVariableData<int64_t> const& b) { return a.variable < b.variable; } );
            std::sort(rationalVariableInformation.begin(), rationalVariableInformation.end(), [] (TransientVariableData<ValueType> const& a, TransientVariableData<ValueType> const& b) { return a.variable < b.variable; } );
        }
        
        template<typename ValueType>
        void TransientVariableInformation<ValueType>::setDefaultValuesInEvaluator(storm::expressions::ExpressionEvaluator<ValueType>& evaluator) const {
            for (auto const& variableData : booleanVariableInformation) {
                evaluator.setBooleanValue(variableData.variable, variableData.defaultValue);
            }
            for (auto const& variableData : integerVariableInformation) {
                evaluator.setIntegerValue(variableData.variable, variableData.defaultValue);
            }
            for (auto const& variableData : rationalVariableInformation) {
                evaluator.setRationalValue(variableData.variable, variableData.defaultValue);
            }
        }
        
        template struct TransientVariableInformation<double>;
        template struct TransientVariableInformation<storm::RationalNumber>;
        template struct TransientVariableInformation<storm::RationalFunction>;
        
    }
}

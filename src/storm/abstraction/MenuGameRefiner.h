#pragma once

#include <functional>
#include <vector>
#include <memory>

#include "storm/abstraction/RefinementCommand.h"
#include "storm/abstraction/QualitativeResultMinMax.h"
#include "storm/abstraction/QuantitativeResultMinMax.h"

#include "storm/storage/expressions/Expression.h"
#include "storm/storage/expressions/PredicateSplitter.h"
#include "storm/storage/expressions/EquivalenceChecker.h"

#include "storm/storage/dd/DdType.h"

#include "storm/utility/solver.h"

namespace storm {
    namespace abstraction {

        template <storm::dd::DdType Type, typename ValueType>
        class MenuGameAbstractor;
        
        template <storm::dd::DdType Type, typename ValueType>
        class MenuGame;
        
        template<storm::dd::DdType Type, typename ValueType>
        class MenuGameRefiner {
        public:
            /*!
             * Creates a refiner for the provided abstractor.
             */
            MenuGameRefiner(MenuGameAbstractor<Type, ValueType>& abstractor, std::unique_ptr<storm::solver::SmtSolver>&& smtSolver);
            
            /*!
             * Refines the abstractor with the given set of predicates.
             */
            void refine(std::vector<storm::expressions::Expression> const& predicates) const;
            
            /*!
             * Refines the abstractor based on the qualitative result by trying to derive suitable predicates.
             *
             * @param True if predicates for refinement could be derived, false otherwise.
             */
            bool refine(storm::abstraction::MenuGame<Type, ValueType> const& game, storm::dd::Bdd<Type> const& transitionMatrixBdd, QualitativeResultMinMax<Type> const& qualitativeResult) const;
            
            /*!
             * Refines the abstractor based on the quantitative result by trying to derive suitable predicates.
             *
             * @param True if predicates for refinement could be derived, false otherwise.
             */
            bool refine(storm::abstraction::MenuGame<Type, ValueType> const& game, storm::dd::Bdd<Type> const& transitionMatrixBdd, QuantitativeResultMinMax<Type, ValueType> const& quantitativeResult) const;
            
        private:
            std::pair<storm::expressions::Expression, bool> derivePredicateFromDifferingChoices(storm::dd::Bdd<Type> const& pivotState, storm::dd::Bdd<Type> const& player1Choice, storm::dd::Bdd<Type> const& lowerChoice, storm::dd::Bdd<Type> const& upperChoice) const;
            std::pair<storm::expressions::Expression, bool> derivePredicateFromPivotState(storm::abstraction::MenuGame<Type, ValueType> const& game, storm::dd::Bdd<Type> const& pivotState, storm::dd::Bdd<Type> const& minPlayer1Strategy, storm::dd::Bdd<Type> const& minPlayer2Strategy, storm::dd::Bdd<Type> const& maxPlayer1Strategy, storm::dd::Bdd<Type> const& maxPlayer2Strategy) const;
            
            /*!
             * Preprocesses the predicates.
             */
            std::vector<storm::expressions::Expression> preprocessPredicates(std::vector<storm::expressions::Expression> const& predicates, bool allowSplits) const;
            
            /*!
             * Creates a set of refinement commands that amounts to splitting all player 1 choices with the given set of predicates.
             */
            std::vector<RefinementCommand> createGlobalRefinement(std::vector<storm::expressions::Expression> const& predicates) const;
            
            void performRefinement(std::vector<RefinementCommand> const& refinementCommands) const;
            
            /// The underlying abstractor to refine.
            std::reference_wrapper<MenuGameAbstractor<Type, ValueType>> abstractor;
            
            /// A flag indicating whether predicates shall be split before using them for refinement.
            bool splitPredicates;

            /// A flag indicating whether predicates shall be split before using them for refinement.
            bool splitGuards;

            /// An object that can be used for splitting predicates.
            mutable storm::expressions::PredicateSplitter splitter;
            
            /// An object that can be used to determine whether predicates are equivalent.
            mutable storm::expressions::EquivalenceChecker equivalenceChecker;
        };
        
    }
}

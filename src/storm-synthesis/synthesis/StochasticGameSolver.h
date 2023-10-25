#pragma once

#include "storm/solver/GameSolver.h"
#include "storm/environment/solver/GameSolverEnvironment.h"

namespace storm {
    namespace synthesis {

        template<typename ValueType>
        class StochasticGameSolver {

        public:
            
            StochasticGameSolver();
            
            void solve(
                std::vector<std::vector<int32_t>> state_to_choice_label_indices,
                storm::storage::SparseMatrix<ValueType> player2_matrix,
                std::vector<ValueType> player2_row_rewards,
                bool player1_maximizing,
                bool player2_maximizing
            );

            std::vector<ValueType> player1_state_values;
            std::vector<uint64_t> player1_choices;
            std::vector<uint64_t> player2_choices;
        
        private:
        
            /** Solver environmert. */
            storm::Environment env;
        };
    }
}
#include "storm-synthesis/synthesis/StochasticGameSolver.h"

namespace storm {
    namespace synthesis {
            
        template<typename ValueType>
        StochasticGameSolver<ValueType>::StochasticGameSolver() {
            env.solver().game().setMethod(storm::solver::GameMethod::ValueIteration);
            env.solver().game().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-8));
        }


        template<typename ValueType>
        void StochasticGameSolver<ValueType>::solve(
            std::vector<std::vector<int32_t>> state_to_choice_label_indices,
            storm::storage::SparseMatrix<ValueType> player2_matrix,
            std::vector<ValueType> player2_row_rewards,
            bool player1_maximizing,
            bool player2_maximizing
        ) {
            // build player1's matrix
            storm::storage::SparseMatrixBuilder<storm::storage::sparse::state_type> player1_matrix_builder(0, 0, 0, false, true);
            uint64_t num_states = state_to_choice_label_indices.size();
            uint64_t player1_num_rows = 0;
            for(uint64_t state=0; state < num_states; state++) {
                player1_matrix_builder.newRowGroup(player1_num_rows);
                for(auto label: state_to_choice_label_indices[state]) {
                    player1_matrix_builder.addNextValue(player1_num_rows,player1_num_rows,1);
                    player1_num_rows++;
                }
            }
            storm::storage::SparseMatrix<storm::storage::sparse::state_type> player1_matrix = player1_matrix_builder.build();
            
            // solve the game
            storm::solver::GameSolverFactory<ValueType> factory;
            auto solver = factory.create(env, player1_matrix, player2_matrix);

            this->player1_state_values.resize(player1_matrix.getRowGroupCount());
            this->player1_choices.resize(player1_matrix.getRowGroupCount());
            this->player2_choices.resize(player2_matrix.getRowGroupCount());

            auto player1_direction = player1_maximizing ? storm::OptimizationDirection::Maximize : storm::OptimizationDirection::Minimize;
            auto player2_direction = player2_maximizing ? storm::OptimizationDirection::Maximize : storm::OptimizationDirection::Minimize;
            solver->solveGame(
                this->env, player1_direction, player2_direction,
                this->player1_state_values, player2_row_rewards,
                &(this->player1_choices), &(this->player2_choices)
            );
        }

        template class StochasticGameSolver<double>;

    }
}
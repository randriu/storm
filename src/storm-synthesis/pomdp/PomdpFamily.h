#pragma once

#include "storm/storage/prism/Program.h"
#include "storm/models/sparse/Model.h"
#include "storm/models/sparse/Pomdp.h"
#include "storm/storage/BitVector.h"

namespace storm {
    namespace synthesis {

        template<typename ValueType>
        class ObservationEvaluator {

        public:

            ObservationEvaluator(
                storm::prism::Program & prism,
                storm::models::sparse::Model<ValueType> const& model
            );

            /** Number of observation expressions. */
            uint32_t num_obs_expressions;
            /** For each observation expression its label. */
            std::vector<std::string> obs_expr_label;
            /** For each observation expression whether it is boolean. */
            std::vector<bool> obs_expr_is_boolean;
            
            /** Number of observation classes. */
            uint32_t num_obs_classes = 0;
            /** For each state its observation class. */
            std::vector<uint32_t> state_to_obs_class;

            /** Get the value of the observation expression in the given observation class. */
            uint32_t observationClassValue(uint32_t obs_class, uint32_t obs_expr);

            /**
             * Create a sub-POMDP from the given sub-MDP by associating its states with observations.
             * @param mdp a sub-MDP
             * @param state_sub_to_full for each state of the sub-MDP the index of the original state
             */
            std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> addObservationsToSubMdp(
                storm::models::sparse::Mdp<ValueType> const& sub_mdp,
                std::vector<uint64_t> state_sub_to_full
            );

            // TODO observation valuations

        private:
            /** Bitwidth of observation expression value size. */
            static const int OBS_EXPR_VALUE_SIZE = 64;
            /** Mapping of observation expressions evaluation to a unique observation class. */
            std::map<storm::storage::BitVector,uint32_t> obs_evaluation_to_class;
            /** Mapping of observation class to observation expressions evaluation. */
            std::vector<storm::storage::BitVector> obs_class_to_evaluation;
            
        };


        template<typename ValueType>
        class QuotientPomdpManager {

        public:

            QuotientPomdpManager(
                storm::models::sparse::Model<ValueType> const& quotient,
                std::vector<uint32_t> state_to_obs_class,
                uint64_t num_actions,
                std::vector<uint64_t> choice_to_action
            );

            /**
             * Create a product of the quotient POMDP and the given FSC.
             * @param num_nodes number of nodes of the FSC
             * @param action_function for each node in the FSC and for each observation class, an index of the choice
             * @param action_function for each node in the FSC and for each observation class, a memory update
             */
            void makeProductWithFsc(
                uint64_t num_nodes,
                std::vector<std::vector<uint64_t>> action_function,
                std::vector<std::vector<uint64_t>> update_function
            );

            // Results of the product construction
            /** For each product state, its state-memory value. */
            std::vector<std::pair<uint64_t,uint64_t>> product_state_to_state_memory;
            /** For aeach product choice, its choice-memory value. */
            std::vector<std::pair<uint64_t,uint64_t>> product_choice_to_choice_memory;
            /** The product. */
            std::shared_ptr<storm::models::sparse::Mdp<ValueType>> product;
            
            /** For each choice of the product MDP, its original choice. */
            std::vector<uint64_t> choice_product_to_original;

            

        private:
            
            /** The quotient model. */
            storm::models::sparse::Model<ValueType> const& quotient;
            /** For each state of the quotient, its observation class. */
            std::vector<uint32_t> state_to_obs_class;
            /** Overall number of actions. */
            uint64_t num_actions;
            /** For each choice of the quotient, the corresponding action. */
            std::vector<uint64_t> choice_to_action;
            
            /** For each state-action pair, a list of choices that implement this action. */
            std::vector<std::vector<std::set<uint64_t>>> state_action_choices;
            /** For each choice, a list of destinations. */
            std::vector<std::set<uint64_t>> choice_destinations;


            /** Number of states in the product. */
            uint64_t productNumberOfStates();
            /** Number of states in the product. */
            uint64_t productNumberOfChoices();

            /** For each state-memory pair, whether it has been registered. */
            std::vector<storm::storage::BitVector> state_memory_registered;
            /** For each state-memory pair, its corresponding product state. */
            std::vector<std::vector<uint64_t>> state_memory_to_product_state;
            /** For each product state, its first choice. */
            std::vector<uint64_t> product_state_row_group_start;
            
            /** Given a state-memory pair, retreive the corresponding product state or create a new one. */
            uint64_t mapStateMemory(uint64_t state, uint64_t memory);
            
            void buildStateSpace(
                uint64_t num_nodes,
                std::vector<std::vector<uint64_t>> action_function,
                std::vector<std::vector<uint64_t>> update_function
            );
            storm::models::sparse::StateLabeling buildStateLabeling();
            storm::storage::SparseMatrix<ValueType> buildTransitionMatrix();
            storm::models::sparse::ChoiceLabeling buildChoiceLabeling();
            storm::models::sparse::StandardRewardModel<ValueType> buildRewardModel(
                storm::models::sparse::StandardRewardModel<ValueType> const& reward_model
            );

            void clearMemory();

        };

    }
}
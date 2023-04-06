#pragma once

#include "storm/models/sparse/Pomdp.h"
#include "storm/logic/Formula.h"

namespace storm {
    namespace synthesis {

        class SubPomdpBuilder {

        public:

            /**
             * Prepare sub-POMDP construction wrt a given canonic POMDP. New
             * sub-POMDP will be model checked using property
             * R[reward_name]=? [F target_label].
             */
            SubPomdpBuilder(
                storm::models::sparse::Pomdp<double> const& pomdp,
                std::string const& reward_name,
                std::string const& target_label
            );

            /**
             * If <1 discount factor is set, each action will redirect 1-df probability to the (target) sink state.
             */
            void setDiscountFactor(double discount_factor) {
                this->discount_factor = discount_factor;
            }

            /** Set which observations to keep in the restricted sub-POMDP. */
            void setRelevantObservations(
                storm::storage::BitVector const& relevant_observations,
                std::map<uint64_t,double> const& initial_belief
            );

            /** Set which states to keep in the restricted sub-POMDP. */
            void setRelevantStates(storm::storage::BitVector const& relevant_states);

            /**
             * Construct a POMDP restriction containing relevant states, frontier states, a new initial state to
             * simulate initial distribution and a new sink state (labeled as a target one) to which frontier states
             * are redirected. Frontier state actions will have reward 0.
             * @param initial_belief initial probability distribution
             * @return a POMDP
             */
            std::shared_ptr<storm::models::sparse::Pomdp<double>> restrictPomdp(
                std::map<uint64_t,double> const& initial_belief
            );

            // observations relevant for the current restriction
            storm::storage::BitVector relevant_observations;
            // states relevant for the current restriction
            storm::storage::BitVector relevant_states;
            // irrelevant states reachable from the relevant ones in one step
            storm::storage::BitVector frontier_states;

            // for each state of a sub-POMDP its index in full POMDP; first two entries corresponding to two fresh
            // states contain 0
            std::vector<uint64_t> state_sub_to_full;
            // for each state of a full POMDP its index in the sub-POMDP, 0 for unreachable states
            // no ambiguity since 0th state in the sub-POMDP is a special state that simulates initial beluef
            std::vector<uint64_t> state_full_to_sub;

        private:

            // original POMDP
            storm::models::sparse::Pomdp<double> const& pomdp;
            // name of the investigated reward
            std::string const reward_name;
            // label assigned to target states
            std::string const target_label;
            // for each state, a list of immediate successors (excluding state itself)
            std::vector<std::set<uint64_t>> reachable_successors;
            // discount factor to be applied to the transformed POMDP
            double discount_factor = 1;
            

            // index of the new initial state
            const uint64_t initial_state = 0;
            // index of the new sink state
            const uint64_t sink_state = 1;
            // label associated with initial distribution as well as shortcut actions
            const std::string empty_label = "";
            
            // total number of states in the sub-POMDP
            uint64_t num_states() {
                return this->relevant_states.getNumberOfSetBits() + this->frontier_states.getNumberOfSetBits() + 2;
            }
            
            
            void constructStateMaps();
            storm::storage::SparseMatrix<double> constructTransitionMatrix(
                std::map<uint64_t,double> const& initial_belief
            );
            storm::models::sparse::StateLabeling constructStateLabeling();
            storm::models::sparse::ChoiceLabeling constructChoiceLabeling(uint64_t num_rows);
            std::vector<uint32_t> constructObservabilityClasses();
            storm::models::sparse::StandardRewardModel<double> constructRewardModel(uint64_t num_rows);
        

        };

    }
}
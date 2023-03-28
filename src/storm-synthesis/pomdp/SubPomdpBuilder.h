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

            /** Set which states to keep in the restricted sub-POMDP. */
            void setRelevantStates(storm::storage::BitVector const& relevant_states);

            /** Get irrelevant states reachable from relevant ones in 1 step. */
            storm::storage::BitVector const& getFrontierStates();

            /**
             * Construct a POMDP restriction containing relevant states, frontier states, a new initial state to
             * simulate initial distribution and a new sink state (labeled as a target one) to which frontier states
             * are redirected.
             * @param initial_belief initial probability distribution
             * @param frontier_values reward obtained upon redirection of the frontier state to the sink state
             * @return a POMDP
             */
            std::shared_ptr<storm::models::sparse::Pomdp<double>> restrictPomdp(
                std::map<uint64_t,double> const& initial_belief,
                std::map<uint64_t,double> const& frontier_values
            );

        private:

            // original POMDP
            storm::models::sparse::Pomdp<double> const& pomdp;
            // name of the investigated reward
            std::string const reward_name;
            // label assigned to target states
            std::string const target_label;
            // for each state, a list of immediate successors (excluding state itself)
            std::vector<std::set<uint64_t>> reachable_successors;
            

            // index of the new initial state
            const uint64_t initial_state = 0;
            // index of the new sink state
            const uint64_t sink_state = 1;
            // label associated with initial distribution as well as shortcut actions
            const std::string empty_label = "";

            // states relevant for the current restriction
            storm::storage::BitVector relevant_states;
            // irrelevant states reachable from the relevant ones in one step
            storm::storage::BitVector frontier_states;
            // for each state of a full POMDP its index in the sub-POMDP, 0 for unreachable states
            // no ambiguity since 0th state in the sub-POMDP is a special states that  simulates initial beluef
            std::vector<uint64_t> state_full_to_sub;

            
            // total number of states in the sub-POMDP
            uint64_t num_states() {
                return this->relevant_states.getNumberOfSetBits() + this->frontier_states.getNumberOfSetBits() + 2;
            }
            
            
            storm::storage::SparseMatrix<double> constructTransitionMatrix(
                std::map<uint64_t,double> const& initial_belief
            );
            storm::models::sparse::StateLabeling constructStateLabeling();
            storm::models::sparse::ChoiceLabeling constructChoiceLabeling(uint64_t num_rows);
            std::vector<uint32_t> constructObservabilityClasses();
            storm::models::sparse::StandardRewardModel<double> constructRewardModel(
                uint64_t num_rows,
                std::map<uint64_t,double> const& frontier_values
            );
        

        };

    }
}
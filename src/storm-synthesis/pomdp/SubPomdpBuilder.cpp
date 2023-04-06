#include "storm-synthesis/pomdp/SubPomdpBuilder.h"

#include "storm/exceptions/InvalidArgumentException.h"

#include "storm/storage/sparse/ModelComponents.h"
#include "storm/storage/SparseMatrix.h"
#include "storm/models/sparse/StandardRewardModel.h"

#include <stack>

namespace storm {
    namespace synthesis {

        
        SubPomdpBuilder::SubPomdpBuilder(
            storm::models::sparse::Pomdp<double> const& pomdp,
            std::string const& reward_name,
            std::string const& target_label
        )
            : pomdp(pomdp), reward_name(reward_name), target_label(target_label) {

            auto const& tm = pomdp.getTransitionMatrix();
            this->reachable_successors.resize(pomdp.getNumberOfStates());
            for(uint64_t state = 0; state < pomdp.getNumberOfStates(); state++) {
                this->reachable_successors[state] = std::set<uint64_t>();
                for(auto const& entry: tm.getRowGroup(state)) {
                    auto successor = entry.getColumn();
                    if(successor != state) {
                        this->reachable_successors[state].insert(successor);
                    }
                }
            }

            this->relevant_states = storm::storage::BitVector(this->pomdp.getNumberOfStates(),false);
            this->frontier_states = storm::storage::BitVector(this->pomdp.getNumberOfStates(),false);
        }

        void SubPomdpBuilder::setRelevantObservations(
            storm::storage::BitVector const& relevant_observations,
            std::map<uint64_t,double> const& initial_belief
        ) {
            this->relevant_observations = relevant_observations;
            this->relevant_states.clear();
            this->frontier_states.clear();

            // traverse the POMDP and identify states with relevant observations that are reachable from the initial belief
            std::stack<uint64_t> state_stack;
            for(const auto &entry : initial_belief) {
                auto state = entry.first;
                this->relevant_states.set(state,true);
                state_stack.push(state);
            }
            while(!state_stack.empty()) {
                auto state = state_stack.top();
                state_stack.pop();
                for(auto dst: this->reachable_successors[state]) {
                    auto dst_obs = this->pomdp.getObservation(dst);
                    if(!this->relevant_observations[dst_obs]) {
                        // dst is a frontier state
                        this->frontier_states.set(dst,true);
                        continue;
                    }
                    // dst is a relevant state
                    if(!this->relevant_states[dst]) {
                        // first encounter of dst
                        this->relevant_states.set(dst,true);
                        state_stack.push(dst);
                    }
                }
            }
        }

        void SubPomdpBuilder::setRelevantStates(storm::storage::BitVector const& relevant_states) {
            this->relevant_states = relevant_states;
            this->frontier_states.clear();
            for(auto state: relevant_states) {
                for(uint64_t successor: this->reachable_successors[state]) {
                    if(!relevant_states[successor]) {
                        this->frontier_states.set(successor,true);
                    }
                }
            }
        }

        void SubPomdpBuilder::constructStateMaps() {
            // create both state maps
            this->state_sub_to_full = std::vector<uint64_t>(this->num_states(),0);
            this->state_full_to_sub = std::vector<uint64_t>(this->pomdp.getNumberOfStates(),0);
            // indices 0 and 1 are reserved for the initial and the sink state respectively
            uint64_t state_index = 2;
            for(auto state: this->relevant_states) {
                this->state_full_to_sub[state] = state_index;
                this->state_sub_to_full[state_index] = state;
                state_index++;
            }
            for(auto state: this->frontier_states) {
                this->state_full_to_sub[state] = state_index;
                this->state_sub_to_full[state_index] = state;
                state_index++;
            }
        }

        
        storm::storage::SparseMatrix<double> SubPomdpBuilder::constructTransitionMatrix(
            std::map<uint64_t,double> const& initial_belief
        ) {
            // num_rows = initial state + sink state + 1 for each frontier state + rows of relevant states
            uint64_t num_rows = 1+1+this->frontier_states.getNumberOfSetBits();
            for(auto state: this->relevant_states) {
                num_rows += this->pomdp.getTransitionMatrix().getRowGroupSize(state);
            }

            // building the transition matrix
            storm::storage::SparseMatrixBuilder<double> builder(
                    num_rows, this->num_states(), 0, true, true, this->num_states()
            );
            uint64_t current_row = 0;

            // initial state distribution
            builder.newRowGroup(current_row);
            for(const auto &entry : initial_belief) {
                auto dst = this->state_full_to_sub[entry.first];
                builder.addNextValue(current_row, dst, entry.second);
            }
            current_row++;

            // sink state self-loop
            builder.newRowGroup(current_row);
            builder.addNextValue(current_row, current_row, 1);
            current_row++;

            // relevant states
            auto const& tm = pomdp.getTransitionMatrix();
            auto const& row_groups = tm.getRowGroupIndices();
            for(auto state: this->relevant_states) {
                builder.newRowGroup(current_row);
                for(uint64_t row = row_groups[state]; row < row_groups[state+1]; row++) {
                    if(this->discount_factor < 1) {
                        builder.addNextValue(current_row, sink_state, 1-this->discount_factor);
                    }
                    for(auto const& entry: tm.getRow(row)) {
                        auto dst = this->state_full_to_sub[entry.getColumn()];
                        builder.addNextValue(current_row, dst, entry.getValue() * this->discount_factor);
                    }
                    current_row++;
                }
            }

            // frontier states are rerouted to the sink state with probability 1
            for(const auto state: this->frontier_states) {
                (void) state;
                builder.newRowGroup(current_row);
                builder.addNextValue(current_row, this->sink_state, 1);
                current_row++;
            }

            // transition matrix finalized
            return builder.build();
        }

        storm::models::sparse::StateLabeling SubPomdpBuilder::constructStateLabeling() {
            // initial state labeling
            storm::models::sparse::StateLabeling labeling(this->num_states());
            storm::storage::BitVector label_init(this->num_states(), false);
            label_init.set(this->initial_state);
            labeling.addLabel("init", std::move(label_init));

            // target state labeling
            storm::storage::BitVector label_target(this->num_states(), false);
            auto const& pomdp_labeling = this->pomdp.getStateLabeling();
            auto const& pomdp_target_states = pomdp_labeling.getStates(this->target_label);
            for(auto state: pomdp_target_states) {
                if(this->relevant_states[state]) {
                    label_target.set(this->state_full_to_sub[state]);
                }
            }
            label_target.set(this->sink_state);
            labeling.addLabel(this->target_label, std::move(label_target));
            
            return labeling;
        }

        storm::models::sparse::ChoiceLabeling SubPomdpBuilder::constructChoiceLabeling(uint64_t num_rows) {
            storm::models::sparse::ChoiceLabeling labeling(num_rows);
            auto const& pomdp_labeling = this->pomdp.getChoiceLabeling();
            labeling.addLabel(this->empty_label, storm::storage::BitVector(num_rows,false));
            for (auto const& label : pomdp_labeling.getLabels()) {
                labeling.addLabel(label, storm::storage::BitVector(num_rows,false));
            }
            uint64_t current_row = 0;

            // initial state, sink state
            labeling.addLabelToChoice(this->empty_label, current_row++);
            labeling.addLabelToChoice(this->empty_label, current_row++);

            // relevant states
            auto const& tm = this->pomdp.getTransitionMatrix();
            auto const& row_groups = tm.getRowGroupIndices();
            for(auto state: this->relevant_states) {
                for(uint64_t row = row_groups[state]; row < row_groups[state+1]; row++) {
                    for(auto label: pomdp_labeling.getLabelsOfChoice(row)) {
                        labeling.addLabelToChoice(label, current_row);
                    }
                    current_row++;
                }
            }

            // frontier states
            for(const auto state: this->frontier_states) {
                (void) state;
                labeling.addLabelToChoice(this->empty_label, current_row++);
            }

            return labeling;
        }

        std::vector<uint32_t> SubPomdpBuilder::constructObservabilityClasses() {
            std::vector<uint32_t> observation_classes(this->num_states());
            uint32_t fresh_observation = this->pomdp.getNrObservations();
            observation_classes[this->initial_state] = fresh_observation;
            observation_classes[this->sink_state] = fresh_observation;
            for(auto state: this->relevant_states) {
                observation_classes[this->state_full_to_sub[state]] = this->pomdp.getObservation(state);
            }
            for(auto state: this->frontier_states) {
                observation_classes[this->state_full_to_sub[state]] = fresh_observation;
            }
            return observation_classes;
        }

        storm::models::sparse::StandardRewardModel<double> SubPomdpBuilder::constructRewardModel(uint64_t num_rows) {
            auto const& reward_model = this->pomdp.getRewardModel(this->reward_name);
            boost::optional<std::vector<double>> state_rewards;
            std::vector<double> action_rewards(num_rows,0);

            uint64_t current_row = 0;

            // skip initial state, sink state
            current_row += 2;

            // relevant states
            auto const& row_groups = this->pomdp.getTransitionMatrix().getRowGroupIndices();
            for(auto state: this->relevant_states) {
                for(uint64_t row = row_groups[state]; row < row_groups[state+1]; row++) {
                    action_rewards[current_row] = reward_model.getStateActionReward(row);
                    current_row++;
                }
            }

            return storm::models::sparse::StandardRewardModel<double>(std::move(state_rewards), std::move(action_rewards));
        }

        std::shared_ptr<storm::models::sparse::Pomdp<double>> SubPomdpBuilder::restrictPomdp(
            std::map<uint64_t,double> const& initial_belief
        ) {
            this->constructStateMaps();
            storm::storage::sparse::ModelComponents<double> components;
            components.transitionMatrix = this->constructTransitionMatrix(initial_belief);
            uint64_t num_rows = components.transitionMatrix.getRowCount();
            components.stateLabeling = this->constructStateLabeling();
            components.choiceLabeling = this->constructChoiceLabeling(num_rows);
            components.observabilityClasses = this->constructObservabilityClasses();
            components.rewardModels.emplace(this->reward_name, this->constructRewardModel(num_rows));
            return std::make_shared<storm::models::sparse::Pomdp<double>>(std::move(components));
        }

    }
}
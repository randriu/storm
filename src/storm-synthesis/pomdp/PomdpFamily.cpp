#include "storm-synthesis/pomdp/PomdpFamily.h"

#include "storm/exceptions/InvalidTypeException.h"
#include "storm/exceptions/NotSupportedException.h"
#include "storm/storage/expressions/ExpressionEvaluator.h"
#include <storm-pomdp/transformer/MakePOMDPCanonic.h>

namespace storm {
    namespace synthesis {

        template<typename ValueType>
        ObservationEvaluator<ValueType>::ObservationEvaluator(
            storm::prism::Program & prism,
            storm::models::sparse::Model<ValueType> const& model
        ) {
            
            // substitute constanst and simplify formulas in the program
            prism = prism.substituteConstantsFormulas(true,true);

            // identify names and types of observation labels
            this->num_obs_expressions = prism.getNumberOfObservationLabels();
            this->obs_expr_label.resize(this->num_obs_expressions);
            this->obs_expr_is_boolean.resize(this->num_obs_expressions);

            for(uint32_t o = 0; o < this->num_obs_expressions; o++) {
                auto const& obs_label = prism.getObservationLabels()[o];
                obs_expr_label[o] = obs_label.getName();
                auto const& obs_expr = obs_label.getStatePredicateExpression();
                STORM_LOG_THROW(obs_expr.hasBooleanType() or obs_expr.hasIntegerType(), storm::exceptions::InvalidTypeException,
                    "expected boolean or integer observation expression");
                this->obs_expr_is_boolean[o] = obs_expr.hasBooleanType();
            }

            // evaluate observation expression for each state valuation
            storm::expressions::ExpressionEvaluator<double> evaluator(prism.getManager());
            auto const& state_valuations = model.getStateValuations();
            // associate each evaluation with the unique observation class
            this->state_to_obs_class.resize(model.getNumberOfStates());
            this->num_obs_classes = 0;
            for(uint64_t state = 0; state < model.getNumberOfStates(); state++) {

                // collect state valuation into evaluator
                for(auto it = state_valuations.at(state).begin(); it != state_valuations.at(state).end(); ++it) {
                    auto const& var = it.getVariable();
                    STORM_LOG_THROW(it.isBoolean() or it.isInteger(), storm::exceptions::InvalidTypeException,
                        "expected boolean or integer variable");
                    // we pass Jani variables to the evaluator, but it seems to work, perhaps it works with variable names
                    if(it.isBoolean()) {
                        evaluator.setBooleanValue(var, it.getBooleanValue());
                    } else if(it.isInteger()) {
                        evaluator.setIntegerValue(var, it.getIntegerValue());
                    }
                }
                
                // evaluate observation expressions and assign class
                storm::storage::BitVector evaluation(OBS_EXPR_VALUE_SIZE*num_obs_expressions);
                for (uint32_t o = 0; o < num_obs_expressions; o++) {
                    evaluation.setFromInt(
                        OBS_EXPR_VALUE_SIZE*o,
                        OBS_EXPR_VALUE_SIZE,
                        evaluator.asInt(prism.getObservationLabels()[o].getStatePredicateExpression())
                    );
                }
                auto result = this->obs_evaluation_to_class.insert(std::make_pair(evaluation,this->num_obs_classes));
                if(not result.second) {
                    // existing evaluation
                    this->state_to_obs_class[state] = result.first->second;
                } else {
                    // new evaluation
                    this->state_to_obs_class[state] = this->num_obs_classes;
                    this->obs_class_to_evaluation.push_back(evaluation);
                    this->num_obs_classes++;
                }
            }
        }

        template<typename ValueType>
        uint32_t ObservationEvaluator<ValueType>::observationClassValue(uint32_t obs_class, uint32_t obs_expr) {
            return this->obs_class_to_evaluation[obs_class].getAsInt(OBS_EXPR_VALUE_SIZE*obs_expr,OBS_EXPR_VALUE_SIZE);
        }

        
        template<typename ValueType>
        std::shared_ptr<storm::models::sparse::Pomdp<ValueType>> ObservationEvaluator<ValueType>::addObservationsToSubMdp(
            storm::models::sparse::Mdp<ValueType> const& sub_mdp,
            std::vector<uint64_t> state_sub_to_full
        ) {

            storm::storage::sparse::ModelComponents<ValueType> components;
            components.transitionMatrix = sub_mdp.getTransitionMatrix();
            components.stateLabeling = sub_mdp.getStateLabeling();
            components.rewardModels = sub_mdp.getRewardModels();
            components.choiceLabeling = sub_mdp.getChoiceLabeling();
            
            std::vector<uint32_t> observability_classes(sub_mdp.getNumberOfStates());
            for(uint64_t state = 0; state < sub_mdp.getNumberOfStates(); state++) {
                observability_classes[state] = this->state_to_obs_class[state_sub_to_full[state]];
            }
            components.observabilityClasses = observability_classes;

            auto pomdp = storm::models::sparse::Pomdp<ValueType>(std::move(components));
            auto pomdp_canonic = storm::transformer::MakePOMDPCanonic<ValueType>(pomdp).transform();
            return pomdp_canonic;
        }

        

        template<typename ValueType>
        QuotientPomdpManager<ValueType>::QuotientPomdpManager(
            storm::models::sparse::Model<ValueType> const& quotient,
            std::vector<uint32_t> state_to_obs_class,
            uint64_t num_actions,
            std::vector<uint64_t> choice_to_action
        ) : quotient(quotient), state_to_obs_class(state_to_obs_class),
            num_actions(num_actions), choice_to_action(choice_to_action) {
            
            this->state_action_choices.resize(this->quotient.getNumberOfStates());
            this->choice_destinations.resize(this->quotient.getNumberOfChoices());
            auto const& row_group_indices = this->quotient.getTransitionMatrix().getRowGroupIndices();
            for(uint64_t state = 0; state < this->quotient.getNumberOfStates(); state++) {
                this->state_action_choices[state].resize(this->num_actions);
                for (uint64_t row = row_group_indices[state]; row < row_group_indices[state+1]; row++) {
                    uint64_t action = this->choice_to_action[row];
                    this->state_action_choices[state][action].insert(row);
                    for(auto const &entry: this->quotient.getTransitionMatrix().getRow(row)) {
                        auto dst = entry.getColumn();
                        this->choice_destinations[row].insert(dst);
                    } 
                }
            }
        }

        
        template<typename ValueType>
        uint64_t QuotientPomdpManager<ValueType>::productNumberOfStates() {
            return this->product_state_to_state_memory.size();
        }

        template<typename ValueType>
        uint64_t QuotientPomdpManager<ValueType>::productNumberOfChoices() {
            return this->product_choice_to_choice_memory.size();
        }
        
        template<typename ValueType>
        uint64_t QuotientPomdpManager<ValueType>::mapStateMemory(uint64_t state, uint64_t memory) {
            if(this->state_memory_registered[state][memory]) {
                return this->state_memory_to_product_state[state][memory];
            }
            auto new_product_state = this->productNumberOfStates();
            this->state_memory_to_product_state[state][memory] = new_product_state;
            this->product_state_to_state_memory.push_back(std::make_pair(state,memory));
            this->state_memory_registered[state].set(memory,true);
            return new_product_state;
        }
        
        template<typename ValueType>
        void QuotientPomdpManager<ValueType>::buildStateSpace(
            uint64_t num_nodes,
            std::vector<std::vector<uint64_t>> action_function,
            std::vector<std::vector<uint64_t>> update_function
        ) {
            this->product_state_to_state_memory.clear();
            this->product_choice_to_choice_memory.clear();
            this->product_state_row_group_start.clear();
            
            uint64_t quotient_num_states = this->quotient.getNumberOfStates();
            this->state_memory_registered.resize(quotient_num_states);
            this->state_memory_to_product_state.resize(quotient_num_states);
            for(uint64_t state = 0; state < quotient_num_states; state++) {
                this->state_memory_registered[state] = storm::storage::BitVector(num_nodes);
                this->state_memory_to_product_state[state].resize(num_nodes);
            }


            uint64_t initial_state = *(this->quotient.getInitialStates().begin());
            uint64_t initial_memory = 0;
            std::queue<uint64_t> unexplored_product_states;
            auto product_state = this->mapStateMemory(initial_state,initial_memory);
            while(true) {
                this->product_state_row_group_start.push_back(this->productNumberOfChoices());
                auto[state,memory] = this->product_state_to_state_memory[product_state];
                auto observation = this->state_to_obs_class[state];
                auto action = action_function[memory][observation];
                auto memory_dst = update_function[memory][observation];
                for(auto choice: this->state_action_choices[state][action]) {
                    this->product_choice_to_choice_memory.push_back(std::make_pair(choice,memory_dst));
                    for(auto state_dst: this->choice_destinations[choice]) {
                        auto dst_registered = this->state_memory_registered[state_dst][memory_dst];
                        auto product_state_dst = this->mapStateMemory(state_dst,memory_dst);
                        if(!dst_registered) {
                            unexplored_product_states.push(product_state_dst);
                        }
                    }
                }
                product_state++;
                if(product_state >= this->productNumberOfStates()) {
                    break;
                }
            }
            this->product_state_row_group_start.push_back(this->productNumberOfChoices());
        }

        
        template<typename ValueType>
        storm::models::sparse::StateLabeling QuotientPomdpManager<ValueType>::buildStateLabeling() {
            storm::models::sparse::StateLabeling product_labeling(this->productNumberOfStates());
            for (auto const& label : this->quotient.getStateLabeling().getLabels()) {
                product_labeling.addLabel(label, storm::storage::BitVector(this->productNumberOfStates(), false));
            }
            for(uint64_t product_state = 0; product_state < this->productNumberOfStates(); product_state++) {
                auto[state,memory] = this->product_state_to_state_memory[product_state];
                for (auto const& label : this->quotient.getStateLabeling().getLabelsOfState(state)) {
                    if(label == "init" and memory != 0) {
                        // init label is only assigned to states with the initial memory state
                        continue;
                    }
                    product_labeling.addLabelToState(label,product_state);
                }
            }
            return product_labeling;
        }

        template<typename ValueType>
        storm::storage::SparseMatrix<ValueType> QuotientPomdpManager<ValueType>::buildTransitionMatrix(
        ) {
            storm::storage::SparseMatrixBuilder<ValueType> builder(0, 0, 0, false, true, 0);
            for(uint64_t product_state = 0; product_state < this->productNumberOfStates(); product_state++) {
                builder.newRowGroup(this->product_state_row_group_start[product_state]);
                for(
                    auto product_choice = this->product_state_row_group_start[product_state];
                    product_choice<this->product_state_row_group_start[product_state+1];
                    product_choice++
                ) {
                    auto[choice,memory_dst] = this->product_choice_to_choice_memory[product_choice];
                    for(auto const &entry: this->quotient.getTransitionMatrix().getRow(choice)) {
                        auto product_dst = this->mapStateMemory(entry.getColumn(),memory_dst);
                        builder.addNextValue(product_choice, product_dst, entry.getValue());
                    }
                }
            }

            return builder.build();
        }

        template<typename ValueType>
        storm::models::sparse::ChoiceLabeling QuotientPomdpManager<ValueType>::buildChoiceLabeling() {
            storm::models::sparse::ChoiceLabeling product_labeling(this->productNumberOfChoices());
            for (auto const& label : this->quotient.getChoiceLabeling().getLabels()) {
                product_labeling.addLabel(label, storm::storage::BitVector(this->productNumberOfChoices(),false));
            }
            for(uint64_t product_choice = 0; product_choice < this->productNumberOfChoices(); product_choice++) {
                auto[choice,memory] = this->product_choice_to_choice_memory[product_choice];
                for (auto const& label : this->quotient.getChoiceLabeling().getLabelsOfChoice(choice)) {
                    product_labeling.addLabelToChoice(label,product_choice);
                }
            }
            return product_labeling;
        }

        template<typename ValueType>
            storm::models::sparse::StandardRewardModel<ValueType> QuotientPomdpManager<ValueType>::buildRewardModel(
                storm::models::sparse::StandardRewardModel<ValueType> const& reward_model
            ) {
                std::optional<std::vector<ValueType>> state_rewards, action_rewards;
                STORM_LOG_THROW(!reward_model.hasStateRewards(), storm::exceptions::NotSupportedException, "state rewards are currently not supported.");
                STORM_LOG_THROW(!reward_model.hasTransitionRewards(), storm::exceptions::NotSupportedException, "transition rewards are currently not supported.");
                
                action_rewards = std::vector<ValueType>();
                for(uint64_t product_choice = 0; product_choice < this->productNumberOfChoices(); product_choice++) {
                    auto[choice,memory] = this->product_choice_to_choice_memory[product_choice];
                    auto reward = reward_model.getStateActionReward(choice);
                    action_rewards->push_back(reward);
                }
                return storm::models::sparse::StandardRewardModel<ValueType>(std::move(state_rewards), std::move(action_rewards));
            }


        template<typename ValueType>
        void QuotientPomdpManager<ValueType>::makeProductWithFsc(
            uint64_t num_nodes,
            std::vector<std::vector<uint64_t>> action_function,
            std::vector<std::vector<uint64_t>> update_function
        ) {

            this->buildStateSpace(num_nodes,action_function,update_function);
            storm::storage::sparse::ModelComponents<ValueType> components;
            components.stateLabeling = this->buildStateLabeling();
            components.transitionMatrix = this->buildTransitionMatrix();
            components.choiceLabeling = this->buildChoiceLabeling();
            for (auto const& reward_model : this->quotient.getRewardModels()) {
                    auto new_reward_model = this->buildRewardModel(reward_model.second);
                    components.rewardModels.emplace(reward_model.first, new_reward_model);
                }
            this->clearMemory();

            this->product = std::make_shared<storm::models::sparse::Mdp<ValueType>>(std::move(components));
        }

        template<typename ValueType>
        void QuotientPomdpManager<ValueType>::clearMemory() {
            this->state_memory_registered.clear();
            this->state_memory_to_product_state.clear();
            this->product_state_row_group_start.clear();
        }


        template class ObservationEvaluator<double>;
        template class QuotientPomdpManager<double>;

    }
}
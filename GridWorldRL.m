%% Get environment dynamics
p = 0.95;
gamma = 0.4;
[NumStates, NumActions, TransitionMatrix, ...
RewardMatrix, StateNames, ActionNames, AbsorbingStates] ...
= PersonalisedGridWorld(p);


%% Compute value function for every state
%unbiased policy P = 1/4 for each action regardless of state
Po = 0.25*ones(1,4);

tol = 0.0001; %tolerance

%obtain value for each state
value = PolicyEvaluation(Po, p, gamma, tol);


%% Gnerate traces using various  policies
unbiased=0.25;
p=0.95;
%Sequence 1
seq1_prob = ((unbiased*p) + ((1-p)/2)*unbiased*2)^4;
%Sequence 2
seq2_prob = ((unbiased*p) + ((1-p)/2)*unbiased*2)^5;
%Sequence 3
seq3_prob = (((unbiased*p) + ((1-p)/2)*unbiased*2)^7) * ((2*unbiased*p)+((unbiased*(1-p)/2)*2));

%unbiased policy for actions N=1, E=2, W = 3
pi1 = zeros(NumStates,NumActions);
pi2 = zeros(NumStates,NumActions);

%Option 1 Policy
pi1(:,1) = 0.5;
pi1(:,2) = 0.2;
pi1(:,3) = 0.1;
pi1(:,4) = 0.2;

%Option 2 Policy
pi2(:,1) = 10/17;
pi2(:,2) = 2/17;
pi2(:,3) = 3/17;
pi2(:,4) = 2/17;

%Get Actions for each state using Policy 1
for s = 1:NumStates
    cumPi = cumsum(pi1(s,:));
    random = rand;
    k=1;
    for k=1:4
        if (random <= cumPi(k))
            act = k;
            break
        end
    end
    newAct1(s) = act;
end

%Get Actions for each state using Policy 2
for s = 1:NumStates
    cumPi = cumsum(pi2(s,:));
    random = rand;
    k=1;
    for k=1:4
        if (random <= cumPi(k))
            act = k;
            break
        end
    end
    newAct2(s) = act;
end
    


%% Generate traces with unbiased policy
%unbiased policy
pi1 = 0.25 * ones(NumStates,NumActions);
episodes = 10;

%obtain MC value and list of traces from policy 
[mean_MC, trace_list] = MC(pi1, p,episodes);



%percent difference between dynamic value and MC value for varying number of episodes
figure
hold on
for ep = 1:50
    [MC_value, trlength] = MC(pi1,p,ep);
    measure = PercentDifference(value,MC_value);
    scatter(ep,measure,'b');
end
title('Value percent difference between dynamic and Monte Carlo methods')
xlabel('Number of traces')
ylabel('Average percent difference (%)')




%% Implement e-Greedy Monte Carlo
%set constants
e1 = 0.1;
e2 = 0.75;
episodes = 50;
trials = 10;

%for each trial compute...
for tr = 1:trials
    [pi1, r1, trace1] = eGreedy_MC(episodes,gamma, e1,p);
    [pi1, r2, trace2] = eGreedy_MC(episodes,gamma, e2,p);
    
    %total rewards for each episode
    reward1(tr,:) = r1
    reward2(tr,:) = r2;
    
    %trace length for each episode
    for l=1:episodes
        rew1 = trace1{l,3}; %1xtrace_length
        rew2 = trace2{l,3}; % 1xtracelength
        
        %get discounted reward for e1 for each episode at each trial
        disc_rew1 = 0;
        for r=1:length(rew1)
            disc_rew1 = gamma^(r-1)*rew1(r)+disc_rew1;
        end
        discR1(tr,l) = disc_rew1;
        
        %get discounted reward for e2 for each episode at each trial
        disc_rew2 = 0;
        for b=1:length(rew2)
            disc_rew2 = gamma^(b-1)*rew2(b)+disc_rew2;
        end
        discR2(tr,l) = disc_rew2;
        
        trace_length1(tr,l) = trace1{l,5};
        trace_length2(tr,l) = trace2{l,5};
    end
    
end
%average and standard deviation of rewards
avg_r1 = mean(discR1);
std_r1 = std(discR1);
avg_r2 = mean(discR2);
std_r2 = std(discR2);

%Used to plot total rewards for each episode
%avg_r1 = mean(reward1);
%std_r1 = std(reward1);
%avg_r2 = mean(reward2);
%std_r2 = std(reward2);

%average and standard deviation of trace length
avg_l1 = mean(trace_length1);
std_l1 = std(trace_length1);
avg_l2 = mean(trace_length2);
std_l2 = std(trace_length2);


%part i plot
figure
hold on
errorbar([1:episodes],avg_r1,std_r1,'LineWidth',2);
errorbar([1:episodes],avg_r2,std_r2,'LineWidth',2);
xlabel('episode');
%ylabel('average reward per episode')
ylabel('discounted reward per episode')
title('Discounted reward vs episodes using 10 trials')
legend('e1 = 0.1', 'e2 = 0.75')
hold off


%part ii plot
figure
hold on
errorbar([1:episodes],avg_l1,std_l1,'LineWidth',2);
errorbar([1:episodes],avg_l2,std_l2,'LineWidth',2);
xlabel('episode');
ylabel('average trace length per episode (limit = 100)')
title('Trace length per episode vs episodes using 20 trials')
legend('e1 = 0.1', 'e2 = 0.75')
hold off




%% PolivyEvaluation Function using dynamic programming
%Computes the value function [V] for states given unbiased policy 'Policy'
function [V] = PolicyEvaluation(Policy, p, gamma, tol)
    [NumStates, NumActions, TransitionMatrix, RewardMatrix, StateNames, ActionNames, AbsorbingStates] = PersonalisedGridWorld(p);
    V = zeros(1,NumStates); %initialize value function
    newV = V;   %value copy
    Delta = 2*tol; 
    count = 0;  %iteration number
    
    %while the difference between succesive values is greater than the
    %tolerance
    while Delta > tol
        for priorState = 1:NumStates
            
            %Do not udate absorbing states
            if (AbsorbingStates(priorState))
            continue;
            end
            
            tmpV = 0;
            %evaluate value for each action given state
            for action = 1:NumActions
                tmpQ = 0;
                for postState = 1:NumStates
                    tmpQ = tmpQ + TransitionMatrix(postState,priorState,action) * (RewardMatrix(postState,priorState,action) + gamma*V(postState));
                end
                %could be Policy(priorState,action)
                tmpV = tmpV + Policy(action)*tmpQ;
            end
            newV(priorState) = tmpV;
        end
        %update for next iteration
        diffVec = abs(newV - V);
        Delta = max(diffVec);
        V = newV;
        count = count+1;
    end
end

%% GetStart function to generate random start state
function start_state = GetStart()
start = round(3*rand)+1;
    output = "";
    switch start
        case 1
            start_state = 11;
        case 2
            start_state = 12;
        case 3
            start_state = 13;
        otherwise 4
            start_state = 14;
    end
end

%% Traces function to generate episode from GridWorld given a policy
%returns a trace which is a list of actions, states, rewards, and String output
function trace = Traces(policy, p)
    [NumStates, NumActions, TransitionMatrix, RewardMatrix, StateNames, ActionNames, AbsorbingStates] = PersonalisedGridWorld(p);
    gamma = 0.4;
    state = GetStart(); %get first state in trace
    output = "S"+state+',';
    post_state(1) = state;
    tr_length = 0;
    trace = {};
    pi = policy(state,:); %policy for current state
    
    
    %get next step in trace as long as state is not absorbing state
    while state ~= 2 || state ~= 3
        tr_length = tr_length+1;
        
        %set limit on trace length to avoid runtime errors
        if tr_length > 75
            break
        end
        
        %get action from policy using random numbers and cumulative
        %probailities
        cumPi = cumsum(pi);
        random = rand;
        for k=1:length(cumPi)
            if (random <= cumPi(k))
                act = k;
                break
            end
        end
        action(tr_length) = ActionNames(act);   %store char of action
        output = output +action(tr_length)+',';
        
        %given state action pair this determines the actual movement
        %relative to the desired direction
        post_probabilities = TransitionMatrix(:,state,act);
        
        %determine next post state using random numbers
        post_CDF = cumsum(post_probabilities);
        random = rand;
        for j=1:length(post_CDF)
            if (random <= post_CDF(j))
                post_state(tr_length+1) = j;
                break;
            end
        end
                
        %Given post state, state, action) we look in matrix to get reward
        reward(tr_length) = RewardMatrix(post_state(tr_length+1), state, act);
        output = output + reward(tr_length)+',';
        
        state = post_state(tr_length+1);    %update for next iteration
     
        %Break loop if next state is absorbing to obtain correct output string
        if state == 2 || state == 3
            break
        end
        output = output + 'S'+state+',';
    end
    output = convertStringsToChars(output);
    output = output(1:end-1); %remove last comma
    tr_length = length(action); %length of trace is length of number of actions
    trace = {post_state, action, reward, output, tr_length}; %return list of everything from episode
   
end



%% Monte-Carlo Policy Evaluation function
%outputs value function for each state and list of traces
function [mean_MC, trace_list] = MC(pi, p,episodes);

    [NumStates, NumActions, TransitionMatrix, RewardMatrix, StateNames, ActionNames, AbsorbingStates] = PersonalisedGridWorld(p);
    value = zeros(1,NumStates); %initialize total value of rewards for each state
    count = zeros(1,NumStates); %initialize number of times state appears
    MC_value = zeros(1,NumStates);
    
    %For each episode
    for ep=1:episodes
        
        %obtain trace lists for given policy
        trace = Traces(pi,p);
        trace_list(ep,:) = trace;
        S = trace{1}; %state list
        A = trace{2}; %action list
        R = trace{3}; %reward list
        O = trace{4}; %output list
        tr_length = trace{5}; %trace length list
        
        %loop through each step in trace
        for i=1:tr_length
        
            %future discounted rewards added to all the previous action state pairs 
            temp = i-1;
            for j=temp:-1:1
                prev_state = S(j);
                value(prev_state) = value(prev_state)+0.4^(i-j)*R(i);
            end
        
            %current reward added to value of current state
            state = S(i);
            value(state) = value(state)+R(i);
            count(state) = count(state)+1;  %track how many times state was visited in trace
        
        end
        MC_value(ep,:) = value./count;    %avg value for each state
    end
    
    %average MC estimates across episodes
    if episodes ~= 1
        mean_MC = nanmean(MC_value);
    else
        mean_MC = MC_value;
    end
end



%% Average Percent Difference
%compute average percent difference of value vs MC_value between all states
function avg_dif = PercentDifference(v1,v2)
value_difference = [];

for c=1:length(v1)
    %check if NaN in MC_value if state not visited
    if isnan(v1(c)) || isnan(v2(c))
        continue
    end
    %percent difference formula
    value_difference(end+1) = (abs(v2(c) - v1(c))/(abs((v2(c) + v1(c))/2))) * 100;
end

%average percent difference
avg_dif = mean(value_difference);

end

%% e-Greedy Monte Carlo function
%returns updated policy, return for each episode, and list of traces for each episode. Input
%paramters include: number of episode, e:[0,1], and  gamma
function [policy,ep_return, trace_list] = eGreedy_MC(episodes,gamma, e,p)

avg_return = 0;
[NumStates, NumActions, TransitionMatrix, RewardMatrix, StateNames, ActionNames, AbsorbingStates] = PersonalisedGridWorld(p);
 
%initialize to unbiased policy each trial
policy = 0.25 * ones(NumStates,NumActions);
    
%Loop through all episodes
for j=1:episodes
    %reset variables for each trial
    Q = zeros(NumStates,NumActions); %average state-action rewards
    Returns = zeros(NumStates,NumActions); %total state-action rewards
    count = zeros(NumStates,NumActions); %count of state-action pairs
 
        
    %start new "game"
    trace = Traces(policy,p);
    trace_list(j,:) = trace;
    S = trace{1};
    Act = trace{2};
    R = trace{3};
    O = trace{4};
    trace_length = trace{5}
    ep_return(j) = sum(R);
        
        
    %loop through enitre trace
    for step=1:trace_length
            
        A = find(ActionNames == Act(step)); %get action number N=1,E=2,S=3,W=4
        state = S(step); %get current state
        count(state,A) = count(state,A)+1; %increment state-action count
        
        G = gamma^(step-1)*R(step); %compute discounted reward from step
            
        %add discounted reward to total state-action reward
        Returns(state,A) = Returns(state,A) + G 
        %Compute average values of state-action pair
        Q(state,A) = Returns(state,A)/count(state,A);
    
        %given the state find which action gives the best action and store
        maxa = (find(Q(state,:) == max(Q(state,:))));
        %if multiple max actions randomly pick one
        maxa = maxa(randi(numel(maxa)));
        
        %update policy in e greedy way given greedy action
        for each_act=1:NumActions
            if (each_act==maxa) %greedy action
                policy(state,each_act) = (1 - e + e/NumActions);
            else %explore
                policy(state,each_act) = e/NumActions;
            end
            
        end
    end
end
    
end


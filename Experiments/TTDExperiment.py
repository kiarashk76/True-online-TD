from Experiments.BaseExperiment import BaseExperiment
from Environments.Chain import Chain
from Agents.TTDAgent import TTDAgent
from Errors.MSVE import MSVE
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import spline

num_states = 19
max_episodes = 50
num_runs = 10
gamma = 0.9
step_size = 2**-3
xxt = 3 #, 1, 37/19
feature_type_list = ['saggregation']#, 'tilecoding', 'binary'] #onehot
convergence_thresh = 1e-100

# step_size_list = np.asarray([i/1000 for i in range(0, int(2000/ xxt), 10)])
# step_size_list = np.asarray([2**(i) for i in range (-20, 2)])
step_size_list = np.asarray([i/1000 for i in range(0, 2500, 100)])
# step_size_list = np.array([2**-16, 2**-14, 2**-12, 2**-10, 2**-6, 2**-4, 2**-2, 2**-1, 1, 1.1, 1.25, 2])
# lamda_list = np.array([1.0])
lamda_list = np.array([1.0, 0.9, 0.8, 0.4, 0.0])
# lamda_list = np.array([0, 0.4, 0.8, 0.9, 1.0])

mode = "parameter_study" #parameter_study
calculate_values = False

def experiment(true_s_values, env, agent, msve, mu, max_episodes= 300, convergence_thresh= 1e-9):

    experiment = BaseExperiment(agent, env)
    err = []
    s_values = np.zeros(num_states)
    for s in range(num_states):
        obs = experiment.observationChannel(s)
        fv = agent.feature_vector(obs)
        s_values[s] = agent.w.dot(fv)
    x = msve.calculate_error(s_values, true_s_values, mu)
    err.append(x)

    for e in range(max_episodes):
        # agent.step_size = step_size / ((e+1)**2) # adaptive step size
        # old_w = np.copy(agent.w)
        experiment.runEpisode()
        # new_w = np.copy(agent.w)
        # dif = old_w - new_w
        # change = dif.dot(dif)
        # if (change < convergence_thresh):
        #     print("converged !", change)
        #     break

        s_values = np.zeros(num_states)
        for s in range(num_states):
            obs = experiment.observationChannel(s)
            fv = agent.feature_vector(obs)
            s_values[s] = agent.w.dot(fv)
        x = msve.calculate_error(s_values, true_s_values, mu)
        err.append(x)

    s_values = np.zeros(num_states)
    for s in range(num_states):
        obs = experiment.observationChannel(s)
        fv = agent.feature_vector(obs)
        s_values[s] = agent.w.dot(fv)
    return err, s_values, agent.w

def plot_err(err_list, name= None):
    # plot the error
    plt.subplots_adjust(hspace= 1)

    for l, lamda in enumerate(lamda_list):
        plt.subplot(len(lamda_list),1,l+1)
        for i, err in enumerate(err_list[l]):
            plt.plot(err, label= str(feature_type_list[i]))
            print(err[-1])

        plt.xlabel('episode_num')
        plt.ylabel('error')

        plt.title( '\u03BB = '+ str(lamda_list[0]) + "\n"+'step size = '+str(step_size))

        plt.legend()
    plt.savefig('../Figures/')
    plt.show()



    # plt.subplots_adjust(hspace= 1)
    #
    # for f, feature_type in enumerate(feature_type_list):
    #     plt.subplot(len(feature_type_list),1,f+1)
    #     for i, err in enumerate(err_list[f]):
    #         plt.plot(err, label= str(lamda_list[i])+' = \u03BB')
    #         print(err[-1])
    #
    #     plt.xlabel('episode_num')
    #     plt.ylabel('error')
    #
    #     plt.title("Representation: "+ feature_type+"\n"+"Step size: "+ str(step_size))
    #
    #     plt.legend()
    # plt.savefig('../Figures/')
    # plt.show()

def plot_parameter(par_list, err_list, par_name, name= None):
    # # 300 represents number of points to make between T.min and T.max
    # for l, lamda in enumerate(lamda_list):
    #     plt.subplot(len(lamda_list), 1, l + 1)
    #     for i, err in enumerate(err_list[l]):
    #         # print(np.var(err), np.mean(err))
    #         # par_list = par_list[err < 2.0]
    #         # err = err[err < 2.0]
    #         # xnew = np.linspace(min(par_list), max(par_list), 50)
    #         # err_smooth = spline(par_list, err, xnew)
    #         # plt.plot(xnew, err_smooth, label= str(lamda_list[i]))
    #         plt.plot(par_list, err, label=str(lamda_list[i]) + ' = \u03BB')
    # plt.xlabel(par_name + "*" + "E[X.XT]")
    # plt.ylabel('error')
    #
    # plt.ylim(top=0.55, bottom=0.0)
    # plt.xlim(left=0.0, right=2.0)
    # plt.legend()
    # plt.savefig('../Figures/')
    # plt.show()


    # 300 represents number of points to make between T.min and T.max
    print(par_list.shape)
    for f, feature_type in enumerate(feature_type_list):
        plt.subplot(len(feature_type_list),1,f+1)
        for i, err in enumerate(err_list[f]):
            # print(np.var(err), np.mean(err))
            # par_list = par_list[err < 2.0]
            # err = err[err < 2.0]
            # xnew = np.linspace(min(par_list), max(par_list), 50)
            # err_smooth = spline(par_list, err, xnew)
            # plt.plot(xnew, err_smooth, label= str(lamda_list[i]))
            # plt.plot(err)
            plt.plot(par_list, err, label= str(lamda_list[i])+' = \u03BB')
            plt.title('state representation: ' + feature_type)

    plt.xlabel(par_name)#+ "*"+"E[X.XT]")
    plt.ylabel('error')

    plt.ylim(top= 0.55, bottom= 0.0)
    plt.xlim(left= 0.0, right= 2.0)
    plt.legend()
    plt.savefig('../Figures/')
    plt.show()

if __name__ == "__main__":
    env = Chain(num_states=num_states)
    actions = env.get_actions()
    agent = TTDAgent(actions=actions, feature_name=feature_type_list[0], num_state=num_states, step_size=step_size,
                     gamma=gamma)

    # calculate the true values
    if calculate_values:
        # for l, lamda in enumerate(lamda_list):
        msve = MSVE(env, 0, agent.policy, gamma)
        # true_s_values = msve.calculate_lreturn_values(np.load('true_values.npy'))
        true_s_values = msve.calculate_true_values()
        mu = msve.calculate_stationary_distribution()
        # print("lambda: ", lamda, true_s_values)
        np.save('true_values', true_s_values)
        np.save('mu', mu)
        # np.save('true_values, lamda=' + str(lamda), true_s_values)  # dump to file

    # running the experiment
    if mode == "parameter_study":
        err_list = np.zeros((len(feature_type_list), len(lamda_list), len(step_size_list)))
    else:
        # err_list = np.zeros((len(feature_type_list), len(lamda_list), max_episodes+1))
        err_list = np.zeros((len(lamda_list), len(feature_type_list), max_episodes+1))


        tot = np.zeros((((len(feature_type_list), len(lamda_list), num_runs , max_episodes+1))))
    for f, feature_type in enumerate(feature_type_list):
        print("state representation: ", feature_type)
        for r in tqdm(range(num_runs)):
            print("start run time number:",r)
            for l, lamda in enumerate(lamda_list):
                true_s_values = np.load('true_values, lamda='+str(lamda)+'.npy')  # load from file
                mu = np.load('mu.npy')
                # true_s_values = np.load('true_values.npy')
                agent = TTDAgent(actions=actions, feature_name=feature_type, num_state=num_states, step_size=step_size,
                                 lamda=lamda, gamma=gamma, seed= r)
                print("testing for lamda: ", lamda)
                agent.lamda = lamda
                msve = MSVE(env, lamda, agent.policy, gamma)

                if mode == "parameter_study":
                    for i, ss in enumerate(step_size_list):
                        agent.step_size = ss
                        err, s_values, w = experiment(true_s_values= true_s_values, env= env, agent= agent, msve= msve,
                                                            max_episodes= max_episodes, convergence_thresh= convergence_thresh, mu= mu)
                        # obj = np.mean(err[-5:])   # err[-1], np.mean(err), np.sum(err)
                        obj = err[-1]
                        # print(w,'->' ,end=" " )
                        # print(ss, np.mean(obj))
                        err_list[f, l, i] += np.mean(obj) / num_runs

                else:
                    err, s_values, w = experiment(true_s_values=true_s_values, env=env, agent=agent, msve=msve,
                                                  max_episodes=max_episodes, convergence_thresh=convergence_thresh, mu= mu)
                    tot[f, l, r] = np.asarray(err)
                    err_list[l, f] += np.asarray(err) / num_runs
                    # err_list[f, l] += np.asarray(err) / num_runs

                    # print("state values: ", s_values, "\n", "weights: ", w)


                print("***")
            print("**********")

        np.save('err_list'+feature_type, err_list)


    if mode == "parameter_study":
        # name = 'ParameterStudy_' + feature_type +',lambda:' + str(lamda_list).replace('.','-')
        par_list = []
        for x in step_size_list:
            par_list.append(x*xxt)
        plot_parameter(par_list=step_size_list, err_list=err_list, par_name="step_size")
    else:
        # name = 'stepsize=' + str(step_size).replace(".", "_ ") + feature_type
        plot_err(err_list)
#
#
#     for l, lamda in enumerate(lamda_list):
#         print("lambda= ", lamda, " var= ", end=' ')
#         print(np.mean(np.var(tot[0][l], axis= 0)))


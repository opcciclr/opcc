import opcc

env_names = [ 'd4rl:maze2d-open-dense-v0',  'd4rl:maze2d-umaze-dense-v1',
              'd4rl:maze2d-medium-dense-v1', 'd4rl:maze2d-large-dense-v1',
              'HalfCheetah-v2', 'Hopper-v2', 'Walker2d-v2']
values = {}
for env in env_names:
    queries = opcc.get_queries(env)

    targets = []
    value_a, value_b = [], []
    confidences = []
    # Batch iteration through Queries :
    for (policy_a_id, policy_b_id), query_batch in queries.items():

        # retrieve policies
        policy_a, policy_b = None, None
        if policy_a_id is not None:
            policy_a, _ = opcc.get_policy(*policy_a_id)
        if policy_b_id is not None:
            policy_b, _ = opcc.get_policy(*policy_b_id)

        # query-a
        value_a += query_batch['info']['return_a']
        value_b += query_batch['info']['return_b']

    values[env] = {'value-a': value_a, 'value-b': value_b}

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
fig, axs = plt.subplots(3,3)
fig.set_size_inches(6, 5)
# gs = gridspec.GridSpec(4, 4)
plot_order = [
    # ['Maze-open', envs_data['d4rl:maze2d-open-v0']],
    # ['Maze-umaze', envs_data['d4rl:maze2d-umaze-v1']],
    # ['Maze-medium', envs_data['d4rl:maze2d-medium-v1']],
    # ['Maze-large', envs_data['d4rl:maze2d-large-v1']],
    ('Maze-open', 'd4rl:maze2d-open-dense-v0'),
    ('Maze-umaze', 'd4rl:maze2d-umaze-dense-v1'),
    ('Maze-medium', 'd4rl:maze2d-medium-dense-v1'),
    ('Maze-large', 'd4rl:maze2d-large-dense-v1'),
    ('HalfCheetah', 'HalfCheetah-v2'),
    ('Hopper', 'Hopper-v2'),
    ('Walker2d', 'Walker2d-v2'),
]
for plot_idx in range(0, 7):
    # if i < 2:
    #     ax = plt.subplot(gs[i//2, 2 * i:2 * i + 2])
    # else:
    #     ax = plt.subplot(gs[i//2, 2 * i - 7:2 * i + 2 - 7])
    title = plot_order[plot_idx][0]
    key = plot_order[plot_idx][1]
    xvalues = values[key]['value-b']
    yvalues = values[key]['value-a']
    i,j = plot_idx//3, plot_idx % 3
    ax = axs[i][j]
    ax.scatter(xvalues, yvalues, marker=".")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(title, fontsize=9)
    if j == 0:
        ax.set_ylabel('$V^{\pi}(s,h)$', fontsize=9)
    if i == 2:
        ax.set_xlabel('$V^{\hat{\pi}}(\hat{s},h)$', fontsize=9)
        ax.set_ylabel('$V^{\pi}(s,h)$', fontsize=9)
    # ax.grid(which='both')
fig.delaxes(axs[2][1])
fig.delaxes(axs[2][2])
plt.tight_layout()
# plt.subplots_adjust(wspace=0.65, hspace=0.55)
plt.savefig('query_viz_all.png', bbox_inches='tight', )
plt.savefig('query_viz_all.eps', bbox_inches='tight', )

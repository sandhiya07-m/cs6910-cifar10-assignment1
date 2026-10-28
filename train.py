
import argparse
import numpy as np
import wandb
from tensorflow.keras.datasets import mnist


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="cs6910-assignment1")
    parser.add_argument("-we", "--wandb_entity", default=None)
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.9)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.999)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0005)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="Xavier")
    parser.add_argument("-nhl", "--num_layers", type=int, default=5)
    parser.add_argument("-sz", "--hidden_size", type=int, default=64)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU")
    return parser.parse_args()


def activate(z, name):
    if name == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    if name == "tanh":
        return np.tanh(z)
    if name == "ReLU":
        return np.maximum(0, z)
    return z


def activation_derivative(z, name):
    if name == "sigmoid":
        a = activate(z, name)
        return a * (1 - a)
    if name == "tanh":
        return 1 - np.tanh(z) ** 2
    if name == "ReLU":
        return (z > 0).astype(float)
    return np.ones_like(z)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(np.clip(z, -50, 50))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def load_data(dataset):
    if dataset == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    else:
        from tensorflow.keras.datasets import fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

    # Fixed validation split: last 10,000 training examples
    x_val, y_val = x_train[-10000:], y_train[-10000:]
    x_train, y_train = x_train[:-10000], y_train[:-10000]

    return x_train, y_train, x_val, y_val, x_test, y_test


def create_model(args, input_size=784, output_size=10):
    sizes = [input_size] + [args.hidden_size] * args.num_layers + [output_size]
    weights, biases = [], []

    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if args.weight_init == "Xavier":
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
        else:
            w = np.random.randn(fan_in, fan_out) * 0.01

        weights.append(w)
        biases.append(np.zeros((1, fan_out)))

    return weights, biases


def forward(x, weights, biases, activation):
    activations = [x]
    zs = []

    for i, (w, b) in enumerate(zip(weights, biases)):
        z = activations[-1] @ w + b
        zs.append(z)

        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = activate(z, activation)

        activations.append(a)

    return activations, zs


def loss_value(y, probs, loss_name):
    one_hot = np.eye(10)[y]
    probs = np.clip(probs, 1e-12, 1.0)

    if loss_name == "cross_entropy":
        return -np.mean(np.sum(one_hot * np.log(probs), axis=1))

    return np.mean(np.sum((probs - one_hot) ** 2, axis=1))


def gradients(x, y, weights, biases, activation, loss_name):
    acts, zs = forward(x, weights, biases, activation)
    batch_size = len(x)
    one_hot = np.eye(10)[y]
    probs = acts[-1]

    if loss_name == "cross_entropy":
        delta = (probs - one_hot) / batch_size
    else:
        delta = (2.0 / batch_size) * (probs - one_hot)
        delta *= probs * (1.0 - probs)

    grad_w = [None] * len(weights)
    grad_b = [None] * len(biases)

    for layer in reversed(range(len(weights))):
        grad_w[layer] = acts[layer].T @ delta
        grad_b[layer] = np.sum(delta, axis=0, keepdims=True)

        if layer > 0:
            delta = (delta @ weights[layer].T)
            delta *= activation_derivative(zs[layer - 1], activation)

    return grad_w, grad_b


def train_step(weights, biases, gw, gb, state, args, step):
    for i in range(len(weights)):
        gw[i] = gw[i] + args.weight_decay * weights[i]

        if args.optimizer == "sgd":
            state["v_w"][i] = gw[i]
            state["v_b"][i] = gb[i]

        elif args.optimizer in ("momentum", "nag"):
            state["v_w"][i] = args.momentum * state["v_w"][i] + gw[i]
            state["v_b"][i] = args.momentum * state["v_b"][i] + gb[i]
            gw[i] = state["v_w"][i]
            gb[i] = state["v_b"][i]

        elif args.optimizer == "rmsprop":
            state["s_w"][i] = args.beta * state["s_w"][i] + (1 - args.beta) * gw[i] ** 2
            state["s_b"][i] = args.beta * state["s_b"][i] + (1 - args.beta) * gb[i] ** 2
            gw[i] = gw[i] / (np.sqrt(state["s_w"][i]) + args.epsilon)
            gb[i] = gb[i] / (np.sqrt(state["s_b"][i]) + args.epsilon)

        else:
            state["m_w"][i] = args.beta1 * state["m_w"][i] + (1 - args.beta1) * gw[i]
            state["m_b"][i] = args.beta1 * state["m_b"][i] + (1 - args.beta1) * gb[i]
            state["s_w"][i] = args.beta2 * state["s_w"][i] + (1 - args.beta2) * gw[i] ** 2
            state["s_b"][i] = args.beta2 * state["s_b"][i] + (1 - args.beta2) * gb[i] ** 2

            mw = state["m_w"][i] / (1 - args.beta1 ** step)
            mb = state["m_b"][i] / (1 - args.beta1 ** step)
            sw = state["s_w"][i] / (1 - args.beta2 ** step)
            sb = state["s_b"][i] / (1 - args.beta2 ** step)

            gw[i] = mw / (np.sqrt(sw) + args.epsilon)
            gb[i] = mb / (np.sqrt(sb) + args.epsilon)

    for i in range(len(weights)):
        weights[i] -= args.learning_rate * gw[i]
        biases[i] -= args.learning_rate * gb[i]


def accuracy(x, y, weights, biases, activation):
    probs = forward(x, weights, biases, activation)[0][-1]
    return float(np.mean(np.argmax(probs, axis=1) == y))


def main():
    args = parse_args()
    np.random.seed(42)

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )

    x_train, y_train, x_val, y_val, x_test, y_test = load_data(args.dataset)
    weights, biases = create_model(args)

    state = {
        key: [np.zeros_like(w) for w in weights]
        for key in ["v_w", "s_w", "m_w"]
    }
    state.update({
        key: [np.zeros_like(b) for b in biases]
        for key in ["v_b", "s_b", "m_b"]
    })

    step = 0

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(x_train))
        x_train, y_train = x_train[indices], y_train[indices]

        for start in range(0, len(x_train), args.batch_size):
            xb = x_train[start:start + args.batch_size]
            yb = y_train[start:start + args.batch_size]

            gw, gb = gradients(
                xb, yb, weights, biases, args.activation, args.loss
            )
            step += 1
            train_step(weights, biases, gw, gb, state, args, step)

        train_probs = forward(x_train[:5000], weights, biases, args.activation)[0][-1]
        val_probs = forward(x_val, weights, biases, args.activation)[0][-1]

        train_loss = loss_value(y_train[:5000], train_probs, args.loss)
        val_loss = loss_value(y_val, val_probs, args.loss)
        train_acc = accuracy(x_train[:5000], y_train[:5000], weights, biases, args.activation)
        val_acc = accuracy(x_val, y_val, weights, biases, args.activation)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"train_accuracy={train_acc:.4f} | val_accuracy={val_acc:.4f}"
        )

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc
        })

    test_acc = accuracy(x_test, y_test, weights, biases, args.activation)
    print(f"Test accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})
    run.finish()


if __name__ == "__main__":
    main()

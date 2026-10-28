
import argparse
import numpy as np
import wandb
from tensorflow.keras.datasets import mnist, fashion_mnist


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-wp", "--wandb_project", default="cs6910-assignment1")
    p.add_argument("-we", "--wandb_entity", default=None)
    p.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="mnist")
    p.add_argument("-e", "--epochs", type=int, default=10)
    p.add_argument("-b", "--batch_size", type=int, default=64)
    p.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    p.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="nadam")
    p.add_argument("-lr", "--learning_rate", type=float, default=0.0001)
    p.add_argument("-m", "--momentum", type=float, default=0.5)
    p.add_argument("-beta", "--beta", type=float, default=0.5)
    p.add_argument("-beta1", "--beta1", type=float, default=0.9)
    p.add_argument("-beta2", "--beta2", type=float, default=0.999)
    p.add_argument("-eps", "--epsilon", type=float, default=1e-8)
    p.add_argument("-w_d", "--weight_decay", type=float, default=0.0005)
    p.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="Xavier")
    p.add_argument("-nhl", "--num_layers", type=int, default=5)
    p.add_argument("-sz", "--hidden_size", type=int, default=64)
    p.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="ReLU")
    return p.parse_args()


def activate(z, name):
    if name == "sigmoid":
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    if name == "tanh":
        return np.tanh(z)
    if name == "ReLU":
        return np.maximum(0, z)
    return z


def derivative(z, name):
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
    loader = mnist if dataset == "mnist" else fashion_mnist
    (x, y), (xt, yt) = loader.load_data()
    x = x.reshape(-1, 784).astype(np.float32) / 255.0
    xt = xt.reshape(-1, 784).astype(np.float32) / 255.0

    # Use 50,000 training examples and 10,000 validation examples.
    xv, yv = x[-10000:], y[-10000:]
    return x[:-10000], y[:-10000], xv, yv, xt, yt


def create_model(args):
    sizes = [784] + [args.hidden_size] * args.num_layers + [10]
    weights, biases = [], []

    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        if args.weight_init == "Xavier":
            limit = np.sqrt(6 / (fan_in + fan_out))
            w = np.random.uniform(-limit, limit, (fan_in, fan_out))
        else:
            w = np.random.randn(fan_in, fan_out) * 0.01
        weights.append(w)
        biases.append(np.zeros((1, fan_out)))

    return weights, biases


def forward(x, weights, biases, activation):
    acts = [x]
    zs = []

    for i, (w, b) in enumerate(zip(weights, biases)):
        z = acts[-1] @ w + b
        zs.append(z)
        a = softmax(z) if i == len(weights) - 1 else activate(z, activation)
        acts.append(a)

    return acts, zs


def loss_value(y, probs, loss_name):
    target = np.eye(10)[y]
    probs = np.clip(probs, 1e-12, 1.0)

    if loss_name == "cross_entropy":
        return float(-np.mean(np.sum(target * np.log(probs), axis=1)))
    return float(np.mean(np.sum((probs - target) ** 2, axis=1)))


def gradients(x, y, weights, biases, activation, loss_name):
    acts, zs = forward(x, weights, biases, activation)
    target = np.eye(10)[y]
    probs = acts[-1]
    n = len(x)

    if loss_name == "cross_entropy":
        delta = (probs - target) / n
    else:
        delta = (2.0 / n) * (probs - target)
        delta *= probs * (1 - probs)

    gw = [None] * len(weights)
    gb = [None] * len(biases)

    for layer in reversed(range(len(weights))):
        gw[layer] = acts[layer].T @ delta
        gb[layer] = np.sum(delta, axis=0, keepdims=True)

        if layer > 0:
            delta = (delta @ weights[layer].T)
            delta *= derivative(zs[layer - 1], activation)

    return gw, gb


def make_state(weights, biases):
    return {
        "vw": [np.zeros_like(v) for v in weights],
        "vb": [np.zeros_like(v) for v in biases],
        "mw": [np.zeros_like(v) for v in weights],
        "mb": [np.zeros_like(v) for v in biases],
        "sw": [np.zeros_like(v) for v in weights],
        "sb": [np.zeros_like(v) for v in biases],
    }


def update(weights, biases, gw, gb, state, args, step):
    for i in range(len(weights)):
        g = gw[i] + args.weight_decay * weights[i]
        h = gb[i] + args.weight_decay * biases[i]

        if args.optimizer == "sgd":
            dw, db = g, h

        elif args.optimizer in ("momentum", "nag"):
            state["vw"][i] = args.momentum * state["vw"][i] + g
            state["vb"][i] = args.momentum * state["vb"][i] + h
            dw, db = state["vw"][i], state["vb"][i]

        elif args.optimizer == "rmsprop":
            state["sw"][i] = args.beta * state["sw"][i] + (1 - args.beta) * g**2
            state["sb"][i] = args.beta * state["sb"][i] + (1 - args.beta) * h**2
            dw = g / (np.sqrt(state["sw"][i]) + args.epsilon)
            db = h / (np.sqrt(state["sb"][i]) + args.epsilon)

        else:
            state["mw"][i] = args.beta1 * state["mw"][i] + (1 - args.beta1) * g
            state["mb"][i] = args.beta1 * state["mb"][i] + (1 - args.beta1) * h
            state["sw"][i] = args.beta2 * state["sw"][i] + (1 - args.beta2) * g**2
            state["sb"][i] = args.beta2 * state["sb"][i] + (1 - args.beta2) * h**2

            mw = state["mw"][i] / (1 - args.beta1**step)
            mb = state["mb"][i] / (1 - args.beta1**step)
            sw = state["sw"][i] / (1 - args.beta2**step)
            sb = state["sb"][i] / (1 - args.beta2**step)

            if args.optimizer == "nadam":
                dw = (args.beta1 * mw + (1 - args.beta1) * g) / (np.sqrt(sw) + args.epsilon)
                db = (args.beta1 * mb + (1 - args.beta1) * h) / (np.sqrt(sb) + args.epsilon)
            else:
                dw = mw / (np.sqrt(sw) + args.epsilon)
                db = mb / (np.sqrt(sb) + args.epsilon)

        weights[i] -= args.learning_rate * dw
        biases[i] -= args.learning_rate * db


def accuracy(x, y, weights, biases, activation):
    probs = forward(x, weights, biases, activation)[0][-1]
    return float(np.mean(np.argmax(probs, axis=1) == y))


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_layers < 1 or args.hidden_size < 1:
        raise ValueError("Epochs, batch size, layers, and hidden size must be positive.")

    np.random.seed(42)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args)
    )

    x, y, xv, yv, xt, yt = load_data(args.dataset)
    weights, biases = create_model(args)
    state = make_state(weights, biases)
    step = 0

    for epoch in range(args.epochs):
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]

        for start in range(0, len(x), args.batch_size):
            xb = x[start:start + args.batch_size]
            yb = y[start:start + args.batch_size]
            gw, gb = gradients(xb, yb, weights, biases, args.activation, args.loss)
            step += 1
            update(weights, biases, gw, gb, state, args, step)

        train_probs = forward(x[:5000], weights, biases, args.activation)[0][-1]
        val_probs = forward(xv, weights, biases, args.activation)[0][-1]

        train_loss = loss_value(y[:5000], train_probs, args.loss)
        val_loss = loss_value(yv, val_probs, args.loss)
        train_acc = accuracy(x[:5000], y[:5000], weights, biases, args.activation)
        val_acc = accuracy(xv, yv, weights, biases, args.activation)

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

    test_acc = accuracy(xt, yt, weights, biases, args.activation)
    print(f"Test accuracy: {test_acc:.4f}")
    wandb.log({"test_accuracy": test_acc})
    run.finish()


if __name__ == "__main__":
    main()

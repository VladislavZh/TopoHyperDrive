# Used the following implementation https://jaketae.github.io/study/pytorch-vgg/
import torch.nn as nn


class MetaSearchVGGLikeWithEmbeddings(nn.Module):
    def __init__(
            self,
            in_channels=3,
            in_height=32,
            in_width=32,
            num_hidden=4096,
            num_classes=100,
            block_1_nb_filter=64,
            block_1_nb_layers=1,
            block_2_nb_filter=128,
            block_2_nb_layers=1,
            block_3_nb_filter=256,
            block_3_nb_layers=1,
            block_4_nb_filter=512,
            block_4_nb_layers=1,
            block_5_nb_filter=512,
            block_5_nb_layers=1,
    ):
        super(MetaSearchVGGLikeWithEmbeddings, self).__init__()

        architecture = (
                [block_1_nb_filter] * block_1_nb_layers + ["M"] +
                [block_2_nb_filter] * block_2_nb_layers + ["M"] +
                [block_3_nb_filter] * block_3_nb_layers + ["M"] +
                [block_4_nb_filter] * block_4_nb_layers + ["M"] +
                [block_5_nb_filter] * block_5_nb_layers
        )

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.convs = self.init_convs(architecture)
        self.fcs = self.init_fcs(architecture)
        self.final_layer = nn.Linear(self.num_hidden, self.num_classes)

    def forward(self, x, return_features=False):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        if return_features:
            return self.final_layer(x), x
        return self.final_layer(x)

    def init_fcs(self, architecture):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        return nn.Sequential(
            nn.Linear(
                last_out_channels * out_height * out_width,
                self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )

    def init_convs(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, int):
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )

        return nn.Sequential(*layers)


class MetaSearchVGGLike(nn.Module):
    def __init__(
            self,
            in_channels=3,
            in_height=32,
            in_width=32,
            num_hidden=4096,
            num_classes=100,
            block_1_nb_filter=64,
            block_1_nb_layers=1,
            block_2_nb_filter=128,
            block_2_nb_layers=1,
            block_3_nb_filter=256,
            block_3_nb_layers=1,
            block_4_nb_filter=512,
            block_4_nb_layers=1,
            block_5_nb_filter=512,
            block_5_nb_layers=1,
    ):
        super(MetaSearchVGGLike, self).__init__()

        architecture = (
                [block_1_nb_filter] * block_1_nb_layers + ["M"] +
                [block_2_nb_filter] * block_2_nb_layers + ["M"] +
                [block_3_nb_filter] * block_3_nb_layers + ["M"] +
                [block_4_nb_filter] * block_4_nb_layers + ["M"] +
                [block_5_nb_filter] * block_5_nb_layers
        )

        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.convs = self.init_convs(architecture)
        self.fcs = self.init_fcs(architecture)

    def forward(self, x, return_features=False):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x

    def init_fcs(self, architecture):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        return nn.Sequential(
            nn.Linear(
                last_out_channels * out_height * out_width,
                self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_classes)
        )

    def init_convs(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if isinstance(x, int):
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )

        return nn.Sequential(*layers)
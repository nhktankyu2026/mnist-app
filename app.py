import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as tvm
from torch.nn.utils import spectral_norm
from torch.optim.swa_utils import AveragedModel
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np


# --- 1. モデル設定の定義 ---
class Config:
    n_classes = 10
    img_size = 32
    latent_dim = 64
    simclr_feat = 512
    simclr_proj = 128
    simclr_comp = 128
    eff_comp = 128
    train_feat = 512
    disc_hidden = 512


cfg = Config()
DEVICE = "cpu"


# --- 2. モデル構造の移植 ---
def sn_linear(i, o, **kw): return spectral_norm(nn.Linear(i, o, **kw))


class Generator(nn.Module):
    def __init__(self,latent_dim=64,n_classes=10,img_ch=3,img_size=32):
        super().__init__()
        self.latent_dim=latent_dim; self.img_size=img_size
        self.embed=nn.Embedding(n_classes,64)
        self.fc=nn.Linear(latent_dim+64,256*4*4)
        self.net=nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.ConvTranspose2d(128,64, 4,2,1),nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4,2,1),nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,img_ch,3,1,1),nn.Tanh())
    def forward(self,z,y):
        e=self.embed(y)
        h=F.relu(self.fc(torch.cat([z,e],1))).view(-1,256,4,4)
        out=self.net(h)
        if out.size(-1)!=self.img_size:
            out=F.interpolate(out,self.img_size,mode="bilinear",align_corners=False)
        return (out+1)/2

class ResNet18Small(nn.Module):
    def __init__(self, feat_dim=512, use_spectral=False):
        super().__init__()
        base = tvm.resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        base.maxpool = nn.Identity()
        self.encoder = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.layer1, base.layer2, base.layer3, base.layer4, base.avgpool)
        if use_spectral:
            for m in self.encoder.modules():
                if isinstance(m, nn.Conv2d): spectral_norm(m)

    def forward(self, x):
        return self.encoder(x).flatten(1)


class EfficientNetB0Features(nn.Module):
    def __init__(self):
        super().__init__()
        eff = tvm.efficientnet_b0(weights=None)
        self.features = eff.features;
        self.pool = eff.avgpool

    def forward(self, x): return self.pool(self.features(x)).flatten(1)


class Discriminator(nn.Module):
    def __init__(self, cfg, simclr_backbone, eff_backbone):
        super().__init__()
        self.simclr_backbone = simclr_backbone;
        self.eff_backbone = eff_backbone
        for p in self.simclr_backbone.parameters(): p.requires_grad_(False)
        for p in self.eff_backbone.parameters():    p.requires_grad_(False)
        self.simclr_comp = nn.Sequential(sn_linear(cfg.simclr_feat, cfg.simclr_comp), nn.LeakyReLU(0.2))
        self.eff_comp = nn.Sequential(sn_linear(1280, cfg.eff_comp), nn.LeakyReLU(0.2))
        self.train_backbone = ResNet18Small(cfg.train_feat, use_spectral=True)
        total = cfg.simclr_comp + cfg.eff_comp + cfg.train_feat
        self.shared = nn.Sequential(
            sn_linear(total, cfg.disc_hidden), nn.LeakyReLU(0.2),
            sn_linear(cfg.disc_hidden, cfg.simclr_proj))
        self.cls_head = sn_linear(cfg.simclr_proj, cfg.n_classes + 1)
        self.embed = nn.Embedding(cfg.n_classes, cfg.simclr_proj)

    def features(self, x):
        with torch.no_grad():
            fs = self.simclr_backbone(x);
            fe = self.eff_backbone(x)
        fs = self.simclr_comp(fs);
        fe = self.eff_comp(fe);
        ft = self.train_backbone(x)
        return self.shared(torch.cat([fs, fe, ft], 1))

    def forward(self, x, y=None):
        h = self.features(x);
        logits = self.cls_head(h)
        if y is not None:
            logits[:, :self.embed.num_embeddings] += (h * self.embed(y)).sum(1, keepdim=True)
        return logits


# --- 3. 両方のモデルをロードする関数 ---
@st.cache_resource
def load_all_models():
    # --- Discriminator (SWA) のロード ---
    simclr_backbone = ResNet18Small(cfg.simclr_feat)
    eff_model = EfficientNetB0Features()
    base_d = Discriminator(cfg, simclr_backbone, eff_model)
    d_model = AveragedModel(base_d)
    d_model.load_state_dict(torch.load("gan_d_swa_final.pth", map_location=DEVICE))
    d_model.eval()

    # --- Generator (EMA) のロード ---
    # ※もしエラーが出る場合は、ノートブックの引数（例: Generator(cfg) など）に合わせてください
    g_model = Generator()
    g_model.load_state_dict(torch.load("gan_g_ema_best.pth", map_location=DEVICE))
    g_model.eval()

    return d_model, g_model


# モデルの初期化
try:
    d_model, g_model = load_all_models()
except NameError:
    st.error("コードの上部に 'Generator' クラスの定義が貼り付けられていないため、起動できません。")
    st.stop()

# --- 4. Web UI 構築 ---
st.title("🤖 🚀 MNIST GAN フル活用Webアプリ")
st.caption("Developed with PyTorch & Streamlit")

# タブの作成
tab1, tab2 = st.tabs(["✍️ 数字を判定する (Discriminator)", "🎨 数字を生み出す (Generator)"])

# ==========================================
# タブ 1: 手書き数字の判別 (D)
# ==========================================
with tab1:
    st.header("手書き数字のリアルタイム判別")
    st.write("キャンバスにマウスや指で「0から9」の数字を描いてみてください。")

    col1, col2 = st.columns([1, 1])
    with col1:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=16,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas_d",
        )

    with col2:
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, :3] > 0):
            img_array = canvas_result.image_data
            img = Image.fromarray(img_array.astype('uint8')).convert('RGB')

            transform = T.Compose([
                T.Resize((cfg.img_size, cfg.img_size)),
                T.ToTensor(),
            ])
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = d_model.module(img_tensor)
                class_logits = logits[:, :cfg.n_classes]
                probs = F.softmax(class_logits, dim=-1)
                pred_class = probs.argmax(dim=-1).item()
                confidence = probs.max(dim=-1).values.item()

            st.markdown(f"<h1 style='text-align: center; font-size: 70px; color: #FF4B4B;'> {pred_class} </h1>",
                        unsafe_allow_html=True)
            st.write(f"**確信度:** {confidence * 100:.2f} %")

            prob_dict = {str(i): float(probs[0][i]) for i in range(10)}
            st.bar_chart(prob_dict)
        else:
            st.info("キャンバスに数字を描いてください。")

# ==========================================
# タブ 2: 数字の自動生成 (G)
# ==========================================
with tab2:
    st.header("AIによる数字の自動画像生成")
    st.write("AI（Generator）に生成させたい数字を選択して、生成ボタンを押してください。")

    # ユーザーに入力させるUI
    target_num = st.selectbox("生成したい数字 (0 ~ 9)", list(range(10)), index=5)

    # ガチャ感を出すための「シード値（乱数）の固定/ランダム」の選択
    random_seed = st.checkbox("毎回ちがう形の数字を作る（ランダム生成）", value=True)

    if st.button("✨ 画像を生成する"):
        # 潜在変数 z とラベル y の準備
        if not random_seed:
            torch.manual_seed(42)  # シードを固定すると同じ画像が再現される

        z = torch.randn(1, cfg.latent_dim, device=DEVICE)
        y = torch.tensor([target_num], dtype=torch.long, device=DEVICE)

        # Generatorで推論（画像生成）
        with torch.no_grad():
            generated_tensor = g_model(z, y)  # 形状: (1, 3, 32, 32)

            # 画像のテンソル値を [-1, 1] から [0, 1] の範囲に引き戻す処理
            # （※Gの最終層が Tanh の場合。もし Sigmoid ならこの処理は不要です）
            generated_tensor = (generated_tensor + 1.0) / 2.0
            generated_tensor = torch.clamp(generated_tensor, 0.0, 1.0)

            # テンソルを numpy 経由で PIL 画像に変換
            img_np = generated_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            gen_img = Image.fromarray(img_np)

        # 画面に引き伸ばして表示 (32x32だと小さすぎるので256ピクセルに拡大表示)
        st.success(f"数字『 {target_num} 』の画像を生成しました！")

        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(gen_img, caption="AIが生成した画像", width=200)
        with col_info:
            st.write("**モデルの裏側での処理:**")
            st.write(f"1. 指定されたラベル `y = {target_num}` を入力")
            st.write(f"2. {cfg.latent_dim}次元のランダムなノイズ `z` をブレンド")
            st.write("3. `gan_g_ema_best.pth` が32x32の擬似画像を演算出力")
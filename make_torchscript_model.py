from source.ppsurf_model import PPSurfModel
from source.ppsurf_data_loader import PPSurfDataModule
import pytorch_lightning as pl

#ckpt = torch.load("test.ckpt", map_location="cpu")
#print(ckpt['state_dict'].keys())

#state_dict = ckpt.get("state_dict", ckpt)
#model = PPSurfModel(pl.LightningModule)
#model.load_state_dict(state_dict)

#model = pl.LightningModule.load_from_checkpoint("test.ckpt", map_location="cpu")
#model.eval()

#model = PPSurfModel.load_from_checkpoint("test2.ckpt")
#model.eval()

#pl.Trainer.load_from_checkpoint("test2.ckpt")
trainer = pl.Trainer(devices=1, accelerator="cpu")
model = PPSurfModel(pointnet_latent_size=256,
                 output_names='imp_surf_sign', in_channels=3, out_channels=2, k=64,
                 lambda_l1=0.0, debug=False, in_file='datasets/abc_minimal/04_pts_vis/00010009_d97409455fa543b3a224250f_trimesh_000.xyz.ply', results_dir='results', padding_factor=0.05, name='ppsurf_mini', network_latent_size=256,
                 gen_subsample_manifold_iter=10, gen_subsample_manifold=10000, gen_resolution_global=257, num_pts_local=50,
                 rec_batch_size=1, gen_refine_iter=1, workers=1)
dataloader = PPSurfDataModule(num_pts_local=50,
                 in_file='datasets/abc_minimal/04_pts_vis/00010009_d97409455fa543b3a224250f_trimesh_000.xyz.ply', workers=1, use_ddp=False, padding_factor=0.05, seed=42, manifold_points=10000,
                 patches_per_shape=100, do_data_augmentation=False, batch_size=1)
try:
  # planned error due to ...
  trainer.test(model, ckpt_path="test.ckpt", datamodule=dataloader)
except Exception:
  pass
  network = trainer.lightning_module
  example_inputs = dict()
  network.to_torchscript(file_path="test-cpu.pt", method='trace', example_inputs=NONE)
#  scripted = torch.jit.script(model)
#  torch.jit.save(scripted, "test-cpu.pt")




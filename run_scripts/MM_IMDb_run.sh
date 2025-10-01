# conda activate multimodal
# HYDRA_FULL_ERROR=1 bash run_scripts/MM_IMDb_run.sh


# no early stopping
for seed in {42..46};do
      python3 main_mmimdb.py \
            seed=${seed} \
            trainer.strategy=ddp_find_unused_parameters_false \
            +model="infmasking" \
            model.model.encoder.embed_dim=768 \
            +data=mmimdb \
            mode="train" \
            +mmimdb.encoders.1.mask_prob=0.15 \
            trainer.max_epochs=70 \
            optim.lr=1e-3 \
            model.model.loss_kwargs.temperature=0.2 \
            model.model.loss_kwargs.mask_lambda=0.5 \
            +model.model.encoder.mask_ratio=0.8 \
            +model.model.encoder.num_mask=4 \
            +model.model.loss_kwargs.weights="[1,1,1,1,5]" \
            +exp_name=seed_${seed}
done



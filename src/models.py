import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l1, l2

REG_PARAMS = {
    "activity_regularizer": l1(0.001),
    "kernel_regularizer": l2(0.01),
    "bias_regularizer": l2(0.01)
}

class Sampling(layers.Layer):
    """Usa (z_mean, z_log_var) per campionare z, il vettore che codifica una cifra."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_encoder(latent_dim=8, dens_layer=512, input_shape=(16, 512, 1)):
    encoder_inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(encoder_inputs)
    x = layers.Conv2D(16, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(32, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)

    # Flatten e Dense
    x = layers.Flatten()(x)
    x = layers.Dense(dens_layer, activation="relu", **REG_PARAMS)(x)

    # Spazio Latente (Media e Log-Varianza)
    z_mean = layers.Dense(latent_dim, name="z_mean", **REG_PARAMS)(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", **REG_PARAMS)(x)
    z = Sampling()([z_mean, z_log_var])

    return keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

def build_decoder(latent_dim=8, dens_layer=512):
    latent_inputs = keras.Input(shape=(latent_dim,))

    # Calcolo dimensione iniziale per il reshape (dipende dagli strides dell'encoder)
    x = layers.Dense(dens_layer, activation="relu", **REG_PARAMS)(latent_inputs)
    x = layers.Dense(1 * 32 * 256, activation="relu", **REG_PARAMS)(x)
    x = layers.Reshape((1, 32, 256))(x)

    # Blocco Convoluzionale Trasposto (Upsampling)
    x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=1, padding="same", **REG_PARAMS)(x)
    x = layers.Conv2DTranspose(16, 3, activation="relu", strides=2, padding="same", **REG_PARAMS)(x)

    # Output Layer (Sigmoid per normalizzazione 0-1)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    return keras.Model(latent_inputs, decoder_outputs, name="decoder")

class VAE(keras.Model):
    def __init__(self, encoder, decoder, alpha=10, beta=2, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.alpha = alpha
        self.beta = beta
        
        # Indici costanti per ON e OFF
        self.ON_IDX = [0, 2, 4]     # A scans
        self.OFF_IDX = [1, 3, 5]    # B scans
        
        # Trackers per le metriche
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.true_loss_tracker = keras.metrics.Mean(name="true_loss")
        self.false_loss_tracker = keras.metrics.Mean(name="false_loss")
        
        # Trackers di validazione
        self.val_total_loss_tracker = keras.metrics.Mean(name="val_total_loss")
        self.val_recon_loss_tracker = keras.metrics.Mean(name="val_recon_loss")
        self.val_kl_loss_tracker = keras.metrics.Mean(name="val_kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.true_loss_tracker,
            self.false_loss_tracker,
            self.val_total_loss_tracker
        ]

    @tf.function
    def loss_diff(self, a, b):
        # FIX: Aggiunto + 1e-7 per evitare divisione per zero (NaN)
        dist_sq = tf.math.reduce_mean(tf.math.reduce_euclidean_norm(a - b, axis=1)**2)
        return 1.0 / (dist_sq + 1e-7)
    
    @tf.function
    def loss_same(self, a, b):
        return tf.math.reduce_mean(tf.math.reduce_euclidean_norm(a - b, axis=1)**2)

    def get_cadence_latents(self, data, training = True): 
        """
        Estrae i vettori latenti 'z' per ciascuno dei 6 step della cadence.
        Mantiene la logica originale: chiama l'encoder separatamente per ogni step.
        """
        latents = []
        # Itera sulle 6 osservazioni della cadence
        for i in range(6):
            # slicing: prende l'i-esima immagine della sequenza temporale
            # data shape: (Batch, 6, H, W, C) -> slice: (Batch, H, W, C)
            current_slice = data[:, i, :, :, :]
            
            # L'encoder restituisce [z_mean, z_log_var, z]. A noi serve z (indice 2)
            z = self.encoder(current_slice, training=training)[2]
            latents.append(z)
        return latents

    @tf.function
    def true_clustering(self, latents):
        """
        Clutering loss per segnali ETI (True)
        Minimizza distanza tra ON-ON e OFF-OFF (loss_same)
        Massimizza distanza tra ON-OFF (loss_diff)
        """
        similarity = 0.0
        
        # Recuperiamo i tensori dalla lista
        z_on = [latents[i] for i in self.ON_IDX]   # [a1, a2, a3]
        z_off = [latents[i] for i in self.OFF_IDX] # [b, c, d]

        # Massimizzare distanza ON vs OFF 
        for z_a in z_on:
            for z_b in z_off:
                similarity += self.loss_diff(z_a, z_b)

        # Minimizzare distanza ON vs ON 
        for i in range(len(z_on)):
            for j in range(len(z_on)):
                if i != j:
                    similarity += self.loss_same(z_on[i], z_on[j])

        # Minimizzare distanza OFF vs OFF 
        for i in range(len(z_off)):
            for j in range(len(z_off)):
                if i != j:
                    similarity += self.loss_same(z_off[i], z_off[j])
        
        return similarity
    
    @tf.function
    def false_clustering(self, latents):
        """
        Clustering loss per segnali non-ETI (False)
        Tutto deve essere simile a tutto 
        """
        similarity = 0.0
        # Confronta ogni vettore con ogni altro vettore nella cadence (6x6)
        for i in range(6):
            for j in range(6):
                if i != j:
                    similarity += self.loss_same(latents[i], latents[j])
        return similarity
        
        
    def train_step(self, data):
        # x è una tupla di 3 input: [Input_Reconstruction, True_Clustering_Cadence, False_Clustering_Cadence]
        x, y = data
        x_recon = x[0]
        true_data = x[1]
        false_data = x[2]
        
        with tf.GradientTape() as tape:
            # Ricostruzione standard (sul primo input)
            z_mean, z_log_var, z = self.encoder(x_recon, training=True)
            reconstruction = self.decoder(z, training=True)
            
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
                )
            )
            
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Calcolo clustering loss
            true_latents = self.get_cadence_latents(true_data, training=True)
            false_latents = self.get_cadence_latents(false_data, training=True)
            
            true_loss = self.true_clustering(true_latents)
            false_loss = self.false_clustering(false_latents)
            
            total_loss = reconstruction_loss + (self.beta * kl_loss) + self.alpha * (true_loss + false_loss)
        
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.true_loss_tracker.update_state(true_loss)
        self.false_loss_tracker.update_state(false_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "true_loss": self.true_loss_tracker.result(),
            "false_loss": self.false_loss_tracker.result()
        }

    def test_step(self, data):
        x, y = data
        x_recon = x[0]
        true_data = x[1]
        false_data = x[2]
        
        z_mean, z_log_var, z = self.encoder(x_recon, training=False)
        reconstruction = self.decoder(z, training=False)
        
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(y, reconstruction), axis=(1, 2)
            )
        )
        
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        
        true_latents = self.get_cadence_latents(true_data, training=False)
        false_latents = self.get_cadence_latents(false_data, training=False)

        true_loss = self.true_clustering(true_latents)
        false_loss = self.false_clustering(false_latents)
        
        total_loss = reconstruction_loss + (self.beta * kl_loss) + self.alpha * (true_loss + false_loss)
        
        self.val_total_loss_tracker.update_state(total_loss)
        self.val_recon_loss_tracker.update_state(reconstruction_loss)
        self.val_kl_loss_tracker.update_state(kl_loss)
        
        return {
            "val_loss": self.val_total_loss_tracker.result(),
            "val_reconstruction_loss": self.val_recon_loss_tracker.result(),
            "val_kl_loss": self.val_kl_loss_tracker.result()
        }

def build_beta_vae(latent_dim=8):
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)

    # Compilazione con ottimizzatore standard
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
    return vae

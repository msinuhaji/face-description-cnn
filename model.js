import * as tf from 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest';

async function main() {
  await tf.ready(); // ensure TF.js is fully loaded
  const x = tf.tensor([1, 2, 3]);
  x.print();
  console.log(x);
}

main();

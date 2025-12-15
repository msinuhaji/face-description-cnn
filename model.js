import * as tf from '@tensorflow/tfjs-node';

async function main() {
  // With tfjs-node tf.ready() is optional, but harmless to await.
  await tf.ready();
  const x = tf.tensor([1, 2, 3]);
  x.print();
  console.log(x);
}

main();

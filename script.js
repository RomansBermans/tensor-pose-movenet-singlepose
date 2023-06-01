const BODY_PARTS = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee', 'left ankle', 'right ankle']

const button = document.getElementById('button')
const image = document.getElementById('image')
const overlay = document.getElementById('overlay')
const stats = document.getElementById('stats')
const info = document.getElementById('info')


const delay = async ms => await new Promise(resolve => setTimeout(resolve, ms))


async function predict(model) {
  const predictionTensor = tf.tidy(() => {
    const imageTensor = tf.browser.fromPixels(image)

    const cropStartPoint = [15, 170, 0];
    const cropSize = [345, 345, 3];
    const croppedTensor = tf.slice(imageTensor, cropStartPoint, cropSize);

    const resizedTensor = tf.image.resizeBilinear(croppedTensor, [192, 192], true).toInt()

    const timeStart = performance.now()
    const predictionTensor = model.predict(tf.expandDims(resizedTensor))
    const modelPredictTime = performance.now() - timeStart
    stats.innerText = `PT: ${Math.round(modelPredictTime)}MS`

    return predictionTensor
  })
  
  const predictions = await predictionTensor.array()
  predictionTensor.dispose()
  
  overlay.replaceChildren()
  predictions[0][0]
    .forEach((prediction, index) => {
      const point = document.createElement('div')
      point.style = `left:${(prediction[1] * 345) + 170}px; top:${(prediction[0] * 345 + 15)}px;`
      point.setAttribute('data-label', `${Math.round(prediction[2] * 100)}% ${BODY_PARTS[index]}`)

      overlay.append(point)
    })
}


async function load() {
  const timeStart = performance.now()
  const model = await tf.loadGraphModel('https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4', { fromTFHub: true })
  const modelLoadTime = performance.now() - timeStart
  
  // TODO:
  // await model.save('localstorage://models/movenet/singlepose/lightning/4')
  // console.log(await tf.io.listModels())
  // await tf.loadLayersModel('localstorage://models/movenet/singlepose/lightning/4')
  
  return { model, modelLoadTime }
}


button.addEventListener('click', async event => {
  event.target.remove()

  stats.innerText = 'Loading Model...'
  const { model, modelLoadTime } = await load()
  
  image.hidden = false
  
  stats.innerText = 'Predicting...'
  await delay(1)
  await predict(model, modelLoadTime)
  
  info.innerText = `LD: ${Math.round(modelLoadTime)}MS MM: ${(tf.memory().numBytes / 1000000).toFixed(2)}MB (${tf.memory().numTensors})`
})
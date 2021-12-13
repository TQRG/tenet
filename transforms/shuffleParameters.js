const utils = require('./lib/utils.js')

function getRandomIndex(arr) {
  const range = arr.length
  return Math.floor(Math.random() * range)
}


function generateShuffledMapping(params) {
  const mappings = {}
  let idxToShuffle = []
  for (let i in params) {
    let p = params[i]
    if (p.type === 'Identifier') {
      idxToShuffle.push(i)
      mappings[i] = i
    }

  }

  // iterate through the idxs
  for (let i of Object.keys(mappings)) {
    let targetIdx = parseInt(i)
    // get a random value index from idxToShuffle that isn't self
    let idxToMapTo = idxToShuffle[getRandomIndex(idxToShuffle)]
    while (idxToMapTo === i && idxToShuffle.length !== 1) {
      idxToMapTo = idxToShuffle[getRandomIndex(idxToShuffle)]
    }
    mappings[i] = idxToMapTo
    // once you select an idx remove it from idxToShuffle
    idxToShuffle = idxToShuffle.filter(e => e !== idxToMapTo)
  }

  return mappings
}


function createNewParams(mapping, params) {
  const oldParams = [...params]
  const newParams = new Array(mapping.length)

  for (let i in params) {
    const idx = parseInt(i)
    if (mapping[i]) {
      const newIdx = parseInt(mapping[i])
      newParams[idx] = oldParams[newIdx]

    } else {
      newParams[idx] = params[idx]
    }
  }

  return newParams
}

module.exports = function(fileInfo, api) {
  const j = api.jscodeshift
  utils.checkValidSource(fileInfo, j)

  let currentSource = j(fileInfo.source)
  const fnParameterMappings = {}
  let calleeParameterNumber = {}

  currentSource.find(j.CallExpression)
    .forEach(p => {
      const calleeName = p.value.callee.name
      calleeParameterNumber[calleeName] = p.value.arguments.length
    })

  // find all fn declarations
  currentSource.find(j.FunctionDeclaration)
    .forEach(p => {
      if (calleeParameterNumber[p.value.id.name] === p.value.params.length) {
        if (p.value.params.length !== 0) {
          const paramMappings = generateShuffledMapping(p.value.params)
          fnParameterMappings[p.value.id.name] = paramMappings
          p.value.params = createNewParams(paramMappings, p.value.params)
        }
      }
      else {
        calleeParameterNumber[p.value.id.name] = -1
      }

      return p
    })

  currentSource.find(j.CallExpression)
    .forEach(p => {
      const calleeName = p.value.callee.name
      if (calleeName && fnParameterMappings[calleeName] && calleeParameterNumber[calleeName] !== -1) {
        const args = p.value.arguments
        p.value.arguments = createNewParams(fnParameterMappings[calleeName], args)
      }

    })

  return currentSource.toSource()
}
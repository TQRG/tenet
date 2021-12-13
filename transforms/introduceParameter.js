const utils = require('./lib/utils.js')

function isSuitableDeclaration(declaration) {
  return declaration.id && declaration.id.type === 'Identifier' && declaration.init && declaration.init.type === 'Literal'
}


function convertDeclarationToParameter(declaration, j) {

  const {id, init} = declaration
  const paramName = id.name
  const paramDefaultVal = init.value

  const paramId = j.identifier(paramName)
  const literal = j.literal(paramDefaultVal)
  return j.assignmentPattern(paramId, literal)

}

module.exports = function(fileInfo, api) {
  const j = api.jscodeshift
  utils.checkValidSource(fileInfo, j)

  let currentSource = j(fileInfo.source)
  let found = false
  let start, end, newParam

  // find the first variable declaration that's appropriate
  const declarations = currentSource.find(j.VariableDeclaration)
    .forEach(p => {
      if (!found && p.value.declarations.length === 1 && isSuitableDeclaration(p.value.declarations[0])) {
        const declBlock = p.value.declarations[0]
        start = p.value.start
        end = p.value.end
        newParam = convertDeclarationToParameter(declBlock, j)
        found = true
      }
    })
  let done = false
  
  if (found) {
    // add the parameter as the last via default
    currentSource.find(j.FunctionDeclaration)
      .forEach(p => {
        if (!done) {
          p.value.params.push(newParam)
          done = true
        }
      })
  }

  if (done) {
    // remove earlier declaration
    const expr = currentSource.find(j.VariableDeclaration, {
      start,
      end
    })
    expr.remove()
  }
  return currentSource.toSource()
}
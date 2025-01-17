from os.path import exists
from loguru import logger
from services import (
    Targets,
    training,
    job
)

def main(filename: str, target: str) -> None:
    try:
        if not exists(filename):
            raise FileExistsError(
                f'The file witn name \'{filename}\' didn\'t find'
            )
        
        if not filename.endswith('.csv'):
            return 'The file has to have .csv extension'

        if target not in [item.value for item in Targets]:
            return 'The target can to be \'training\' or \'job\''
        
        response = training(filename) if target == 'training' else job(filename)
        return response
    
    except (FileExistsError, Exception, ) as e:
        return str(e)

if __name__ == '__main__':
    try:
        filename = str(input('Filename: '))
        target = str(input('Target: '))

        if target:
            target = target.lower()    

        response = main(filename, target)
        logger.debug(response)
    
    except KeyboardInterrupt:
        logger.info('Exit.')

    except Exception as e:
        logger.error(str(e))
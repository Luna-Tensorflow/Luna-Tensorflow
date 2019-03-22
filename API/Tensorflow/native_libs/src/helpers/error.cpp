//Source: https://github.com/luna/Dataframes/blob/7031d5613fb95a2945b3c73226cbcec3d5380c02/native_libs/src/Core/Error.cpp

#include "error.h"

#include <iostream>
#include <utility>
#include "logging.h"

namespace
{
    const auto unknownInternalErrorText = "Unknown internal error encountered";
    thread_local std::string errorMessage;


    void writeError(const char **outError, const char *toWrite)
    {
        if(toWrite)
            LOG("Setting error message: {}", toWrite);

        if(outError)
        {
            *outError = toWrite;
        }
        else if(toWrite)
        {
            // should not happen unless caller gives nullptr as error target
            // in such case we will just write the error to stdout
            // as we have no other means of getting user's attention
            std::cerr << "outError==nullptr, failed to set error message: " << toWrite << std::endl;
        }
    }
}

void setError(const char **outError, const char *errorToSet, const char *functionName) noexcept
{
    try
    {
        if(functionName)
            errorMessage = std::string(functionName) + ": " + errorToSet;
        else
            errorMessage = errorToSet;

        writeError(outError, errorMessage.c_str());
    }
    catch(...)
    {
        // should happen practically never, perhaps if error string is too long to fit in memory....
        // but to be on the safe side
        writeError(outError, unknownInternalErrorText);
    }
}

void clearError(const char **outError) noexcept
{
    writeError(outError, nullptr);
}
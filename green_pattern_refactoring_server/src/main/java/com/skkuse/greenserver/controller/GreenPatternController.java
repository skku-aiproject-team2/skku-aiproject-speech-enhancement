package com.skkuse.greenserver.controller;

import com.skkuse.greenserver.code.Fixed;
import com.skkuse.greenserver.greenpattern.GreenPatternProcessor;
import com.skkuse.greenserver.code.Buggy;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/green-pattern")
@RequiredArgsConstructor
public class GreenPatternController {
    private final GreenPatternProcessor greenPatternProcessor;

    //@ResponseBody
    @PostMapping
    public Fixed processGreenPattern(@RequestBody Buggy buggy) {
        return greenPatternProcessor.getGreenPattern(buggy);
    }
}
